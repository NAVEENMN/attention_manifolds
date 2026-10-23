"""Attention Manifolds: local streaming demo with isolated model interventions."""
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import json
import logging
import os
import queue
import threading
import time
import hashlib
import uuid
import transformers
from typing import Annotated, Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from spline_qk import load_splines, projections

ROOT = Path(__file__).resolve().parent
MODEL = TOKENIZER = None
TRAINED = {}
CALIBRATED = {}
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
STATUS = {'state': 'loading', 'model': 'Llama-3.2-3B-Instruct', 'layers': 28, 'device': DEVICE}
MODEL_LOCK = threading.Lock()
BUILD = hashlib.sha256(b''.join((ROOT / name).read_bytes() for name in
    ('app.py', 'spline_qk.py', 'static/app.js', 'static/index.html'))).hexdigest()[:12]
STATUS['build'] = BUILD
STATUS['baseline'] = 'original_linear_projections'


def checkpoint_digest(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.glob('*.pt')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update((directory / 'manifest.json').read_bytes())
    return digest.hexdigest()


def configure_model(body):
    """Called only while holding MODEL_LOCK; return the applied configuration."""
    selected = set(body.layers)
    scales = {i: body.layer_alphas.get(i, body.alpha) if i in selected else 1.0
              for i in range(STATUS['layers'])}
    with torch.no_grad():
        for key, proj in projections(MODEL).items():
            proj.enabled = body.mode == 'custom'
            value = TRAINED[key]
            alpha = scales[key[0]]
            if body.mode == 'custom' and alpha != 1:
                value = CALIBRATED[key] if alpha == 0 else CALIBRATED[key] + alpha * (value - CALIBRATED[key])
            proj.coefficients.copy_(value)
    return {'mode': body.mode, 'model': STATUS['model'], 'build': BUILD,
            'projection': 'bspline' if body.mode == 'custom' else 'original_linear',
            'layer_alphas': scales if body.mode == 'custom' else {},
            'overrides': {i: scales[i] for i in sorted(selected)} if body.mode == 'custom' else {},
            'max_tokens': body.max_tokens, 'decoding': 'greedy',
            'runtime': STATUS.get('runtime', {})}


def restore_model():
    with torch.no_grad():
        for key, proj in projections(MODEL).items():
            proj.enabled = False
            proj.coefficients.copy_(TRAINED[key])

# Reduce numerical variation; this does not guarantee identical outputs across GPUs.
torch.set_float32_matmul_precision('highest')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_models():
    global MODEL, TOKENIZER
    try:
        trained_dir = Path(os.getenv('MANIFOLDS_CHECKPOINT', str(ROOT / 'models/3b_trained')))
        calibrated_dir = Path(os.getenv('MANIFOLDS_CALIBRATED', str(ROOT / 'models/3b_calibrated')))
        meta = json.loads((trained_dir / 'manifest.json').read_text())['metadata']
        model = AutoModelForCausalLM.from_pretrained(meta['model'], dtype=torch.float32, attn_implementation='eager').eval().to(DEVICE)
        model.requires_grad_(False)
        TOKENIZER = AutoTokenizer.from_pretrained(meta['model'])
        load_splines(trained_dir, model)
        manifest = json.loads((calibrated_dir / 'manifest.json').read_text())
        for entry in manifest['layers']:
            states = torch.load(calibrated_dir / entry['file'], map_location='cpu', weights_only=True)
            for kind in ('q', 'k'):
                key = (entry['layer'], kind)
                proj = projections(model)[key]
                state = states[kind]
                if not torch.equal(proj.center.cpu(), state['center']) or not torch.equal(proj.radius.cpu(), state['radius']):
                    raise ValueError('Checkpoint domains do not match')
                TRAINED[key] = proj.coefficients.detach().clone()
                CALIBRATED[key] = state['coefficients'].to(DEVICE)
                proj.enabled = False
        model.requires_grad_(False)
        MODEL = model
        runtime = {'device': DEVICE, 'dtype': 'float32', 'attention': 'eager',
                   'torch': torch.__version__, 'transformers': transformers.__version__,
                   'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name() if DEVICE == 'cuda' else None,
                   'tf32': torch.backends.cuda.matmul.allow_tf32,
                   'matmul_precision': torch.get_float32_matmul_precision(),
                   'model_revision': getattr(model.config, '_commit_hash', None),
                   'chat_template_sha256': hashlib.sha256(str(TOKENIZER.chat_template).encode()).hexdigest(),
                   'trained_sha256': checkpoint_digest(trained_dir),
                   'calibrated_sha256': checkpoint_digest(calibrated_dir)}
        STATUS.update(state='ready', model=meta['model'].split('/')[-1], layers=len(model.model.layers), runtime=runtime)
    except Exception:
        logging.exception('Model loading failed')
        STATUS.update(state='error', message='Unable to load the model. Check the server log and checkpoint paths.')


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=load_models, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')


@app.middleware('http')
async def prevent_stale_assets(request, call_next):
    response = await call_next(request)
    if request.url.path == '/' or request.url.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Manifolds-Build'] = BUILD
    return response


@app.get('/')
def index():
    return FileResponse(ROOT / 'static/index.html', headers={'Cache-Control': 'no-cache'})


@app.get('/api/status')
def status():
    return {**STATUS, 'busy': MODEL_LOCK.locked()}


class Message(BaseModel):
    role: str = Field(pattern='^(user|assistant)$')
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    mode: Literal['baseline', 'custom'] = 'baseline'
    messages: list[Message] = Field(min_length=1, max_length=1)
    layers: list[int] = Field(default_factory=list, max_length=128)
    alpha: float = Field(default=1, ge=-3, le=3, allow_inf_nan=False)
    layer_alphas: dict[int, Annotated[float, Field(ge=-3, le=3, allow_inf_nan=False)]] = Field(default_factory=dict, max_length=128)
    max_tokens: int = Field(default=256, ge=32, le=512)


class StopRequested(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


@app.post('/api/chat')
async def chat(body: ChatRequest):
    if STATUS['state'] != 'ready':
        raise HTTPException(503, 'The model is still loading. Please try again shortly.')
    if body.messages[-1].role != 'user' or any(m.role != ('user' if i % 2 == 0 else 'assistant') for i, m in enumerate(body.messages)):
        raise HTTPException(422, 'Messages must alternate between user and assistant.')
    if any(i < 0 or i >= STATUS['layers'] for i in body.layers):
        raise HTTPException(422, 'Invalid layer selection.')
    if any(i not in body.layers for i in body.layer_alphas):
        raise HTTPException(422, 'Per-layer scales must refer to selected layers.')
    if sum(len(m.content) for m in body.messages) > 40000:
        raise HTTPException(413, 'This conversation is too long. Please start a new chat.')
    # Fail fast: no hidden queue or shared coefficient mutations across visitors.
    if not MODEL_LOCK.acquire(blocking=False):
        raise HTTPException(429, 'The model is answering another request. Please try again in a moment.')
    stop = threading.Event()
    events = queue.Queue()

    def worker():
        started = time.monotonic()
        try:
            config = configure_model(body)
            events.put({'type': 'configuration', 'request_id': str(uuid.uuid4()), **config})
            logging.info('Applied mode=%s overrides=%s build=%s', body.mode, config['overrides'], BUILD)
            ids = TOKENIZER.apply_chat_template([m.model_dump() for m in body.messages], add_generation_prompt=True, return_tensors='pt')
            if hasattr(ids, 'input_ids'):
                ids = ids['input_ids']
            if ids.shape[-1] > 4096:
                raise ValueError('This conversation exceeds the demo context limit. Please start a new chat.')
            ids = ids.to(DEVICE)
            class Streamer(TextIteratorStreamer):
                def on_finalized_text(self, text, stream_end=False):
                    if text:
                        events.put({'type': 'token', 'text': text})
            streamer = Streamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
            with torch.inference_mode():
                output = MODEL.generate(input_ids=ids, attention_mask=torch.ones_like(ids), streamer=streamer,
                    max_new_tokens=body.max_tokens, do_sample=False, use_cache=True,
                    pad_token_id=TOKENIZER.eos_token_id, stopping_criteria=StoppingCriteriaList([StopRequested(stop)]))
            events.put({'type': 'done', 'seconds': round(time.monotonic()-started, 1),
                        'tokens': output.shape[-1]-ids.shape[-1], 'stopped': stop.is_set(),
                        'truncated': output.shape[-1]-ids.shape[-1] >= body.max_tokens})
        except Exception as exc:
            logging.exception('Generation failed')
            message = str(exc) if isinstance(exc, ValueError) else 'Generation failed. Please try a shorter prompt.'
            events.put({'type': 'error', 'message': message})
        finally:
            try:
                restore_model()
            finally:
                MODEL_LOCK.release()
                events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def stream():
        try:
            yield 'data: ' + json.dumps({'type': 'start'}) + '\n\n'
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(.03)
                    continue
                if event is None:
                    break
                yield 'data: ' + json.dumps(event) + '\n\n'
        finally:
            stop.set()
    return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    import uvicorn
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler('demo.log', mode='w'), logging.StreamHandler()]
    )
    uvicorn.run(app, host='127.0.0.1', port=int(os.getenv('PORT', '7860')))
