use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use pyo3::{IntoPy, PyAny, pyclass, pymethods, PyResult, Python};
use pyo3::types::{PyDict};
use std::thread;
use eyre::{ContextCompat, eyre};
use scopeguard::defer;
use tokio::sync::mpsc::{UnboundedSender, Sender, Receiver};

#[derive(Debug)]
struct PipelineRequest {
    request: CompletionRequest,
    send_response: UnboundedSender<eyre::Result<PipelineResponse>>,
    cancellation_flag: Arc<AtomicBool> // true if cancelled
}

#[derive(Debug)]
struct PipelineResponse(CompletionResponse);

#[derive(Debug)]
pub struct CompletionRequest {
    pub prompt: String,

    pub temperature: f32,
    pub max_new_tokens: u32,
    pub top_p: f32,
    pub repetition_penalty: f32
}

#[derive(Debug)]
pub struct CompletionResponse {
    pub generated_text: String
}

#[derive(Clone)]
pub struct LLMManager {
    send_req: Sender<PipelineRequest>
}

impl LLMManager {
    pub fn init() -> Self {
        let (send_req, recv_req) = tokio::sync::mpsc::channel(1);
        thread::spawn(|| {
            launch_processor(recv_req).unwrap();
        });
        Self { send_req }
    }

    pub async fn inference(&self, request: CompletionRequest) -> eyre::Result<CompletionResponse> {
        let cancellation_flag = Arc::new(AtomicBool::new(false));
        let cancellation_flag_clone = cancellation_flag.clone();
        defer! {
            // Mark request as cancelled if future is dropped
            cancellation_flag_clone.store(true, Ordering::Relaxed);
        }

        let (
            send_response,
            mut recv_response
        ) = tokio::sync::mpsc::unbounded_channel();
        self.send_req.send(PipelineRequest {
            request,
            send_response,
            cancellation_flag
        }).await?;
        recv_response
            .recv()
            .await
            .context("no response received")?
            .map(|r| r.0)
    }
}

fn launch_processor(mut recv_req: Receiver<PipelineRequest>) -> eyre::Result<()> {
    println!("Loading model...");
    Python::with_gil(|py| {
        let locals = PyDict::new(py);
        py.run(
            LOAD_SCRIPT,
            None,
            Some(locals)
        )?;

        println!("Model successfully loaded!");

        loop {
            let Some(req) = recv_req.blocking_recv() else {
                break;
            };

            let result = process_request(py, locals, req.request, &req.cancellation_flag);
            let _ = req.send_response.send( // We don't care if request received or not
                result
                    .map(PipelineResponse)
                    // We convert error to string as it seems to cause a hang if we don't and then try to print the error later
                    // I assume this is because we try to lock the GIL when we print the error
                    .map_err(|e| eyre!("LLM error: {e:?}"))
            );
        }

        Ok(())
    })
}

fn process_request(py: Python, locals: &PyDict, req: CompletionRequest, cancellation_flag: &Arc<AtomicBool>) -> eyre::Result<CompletionResponse> {
    let dict = PyDict::new(py);
    dict.set_item("prompt", &req.prompt)?;
    dict.set_item("max_new_tokens", req.max_new_tokens)?;
    dict.set_item("temperature", req.temperature)?;
    dict.set_item("top_p", req.top_p)?;
    dict.set_item("repetition_penalty", req.repetition_penalty)?;
    let callback = Box::new(CancelCallback {
        cancelled: cancellation_flag.clone()
    });
    dict.set_item("callback", callback.into_py(py))?;
    locals.set_item("req", dict)?;
    let generated_text: String = py.eval(INFER_SCRIPT, None, Some(locals))?.extract()?;
    Ok(CompletionResponse {
        generated_text: req.prompt + &generated_text
    })
}

#[pyclass]
struct CancelCallback {
    cancelled: Arc<AtomicBool>
}

#[pymethods]
impl CancelCallback {
    fn __call__(&self, _step_result: &PyAny) -> PyResult<bool> {
        Ok(self.cancelled.load(Ordering::Relaxed))
    }
}

/*
GPTQ script:
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

use_triton = True

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(model_path,
        use_safetensors=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)
#logging.set_verbosity(logging.DEBUG)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
 */
//language=python
const LOAD_SCRIPT: &str = r#"
model_path = "/data"

from hf_hub_ctranslate2 import GeneratorCT2fromHfHub
from transformers import AutoTokenizer

model = GeneratorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_path,
        device="cuda",
        compute_type="int8",
        #compute_type="int8_float16",
        tokenizer=AutoTokenizer.from_pretrained(model_path + "/tokenizer")
)
"#;

/*
GPTQ script:
pipe(
    req['prompt'],
    max_new_tokens=req['max_new_tokens'],
    do_sample=True,
    temperature=req['temperature'],
    top_p=req['top_p'],
    repetition_penalty=req['repetition_penalty']
)[0]['generated_text']
 */
const INFER_SCRIPT: &str = r#"
model.generate(
    text=[req['prompt']],
    max_length=req['max_new_tokens'],
    sampling_topp=req['top_p'],
    sampling_temperature=req['temperature'],
    repetition_penalty=req['repetition_penalty'],
    callback=req['callback'],
    beam_size=1,
    return_scores=False,
    include_prompt_in_result=False
)[0]
"#;
