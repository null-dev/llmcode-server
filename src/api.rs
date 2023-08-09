use axum::extract::State;
use axum::http::StatusCode;
use axum::{Json, Router};
use axum::routing::post;
use serde::{Serialize, Deserialize};
use crate::llm::{CompletionRequest, LLMManager};

#[derive(Deserialize)]
struct InferenceRequest {
    inputs: String,
    parameters: InferenceParameters
}

#[derive(Deserialize)]
struct InferenceParameters {
    temperature: f32,
    max_new_tokens: u32,
    top_p: f32,
    repetition_penalty: f32
}

type InferenceResponse = Vec<InferenceEntry>;

#[derive(Serialize)]
struct InferenceEntry {
    generated_text: String
}

#[derive(Clone)]
struct APIState {
    llm: LLMManager
}

#[tokio::main]
pub(super) async fn init(llm: LLMManager) -> eyre::Result<()> {
    let app = Router::new()
        .route("/generate", post(generate))
        .with_state(APIState { llm });

    axum::Server::bind(&"0.0.0.0:80".parse()?)
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("expected shutdown signal");
    println!("Shutting down...");
}

async fn generate(
    State(state): State<APIState>,
    Json(payload): Json<InferenceRequest>
) -> (StatusCode, Json<Option<InferenceResponse>>) {
    let converted_req = CompletionRequest {
        prompt: payload.inputs,

        temperature: payload.parameters.temperature,
        max_new_tokens: payload.parameters.max_new_tokens,
        top_p: payload.parameters.top_p,
        repetition_penalty: payload.parameters.repetition_penalty,
    };
    let result = state.llm.inference(converted_req).await;
    match result {
        Ok(resp) => (StatusCode::OK, Json(Some(vec![InferenceEntry {
            generated_text: resp.generated_text
        }]))),
        Err(err) => {
            // Do not block
            let _ = tokio::task::spawn_blocking(move || {
                println!("Inference error: {err:?}");
            }).await;
            (StatusCode::INTERNAL_SERVER_ERROR, Json(None))
        }
    }
}