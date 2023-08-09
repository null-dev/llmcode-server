use crate::llm::LLMManager;

mod api;
mod llm;

fn main() -> eyre::Result<()> {
    let llm = LLMManager::init();
    api::init(llm)
}