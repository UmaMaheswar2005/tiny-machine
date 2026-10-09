use notify::{Config, RecursiveMode, Result, Watcher};
use std::fs;
use std::path::Path;
use std::thread::sleep;
use std::time::Duration;
use uuid::Uuid;
use walkdir::WalkDir;

#[tokio::main]
async fn main() -> Result<()> {
    let qdrant_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "http://localhost:6333".to_string());
    let qdrant_api_key = std::env::var("QDRANT_API_KEY").unwrap_or_default();
    let jina_api_key = std::env::var("JINA_API_KEY").unwrap_or_default();

    let path_to_watch = "./data";
    println!("🚀 Hydra-Ingestor: Watching folder: {}", path_to_watch);

    let client = reqwest::Client::new();

    // 1. Initialize Qdrant Collection (768 dimensions for Jina Embeddings)
    let qdrant_init = serde_json::json!({
        "vectors": { "size": 768, "distance": "Cosine" }
    });

    let mut qdrant_init_req = client.put(format!("{}/collections/hydra_docs", qdrant_url));
    if !qdrant_api_key.is_empty() {
        qdrant_init_req = qdrant_init_req.header("api-key", &qdrant_api_key);
    }
    let _ = qdrant_init_req.json(&qdrant_init).send().await;
    println!("🗄️ Database: 'hydra_docs' collection ready at {}", qdrant_url);

    // 2. INITIAL BOOT SCAN
    println!("🔍 Performing initial scan of existing files in {}...", path_to_watch);
    for entry in WalkDir::new(path_to_watch).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            process_file(path, &client, &qdrant_url, &qdrant_api_key, &jina_api_key).await;
        }
    }

    // 3. Setup watcher for real-time file additions
    let (tx, mut rx) = tokio::sync::mpsc::channel(100);
    let mut watcher = notify::RecommendedWatcher::new(
        move |res| {
            let _ = tx.blocking_send(res);
        },
        Config::default(),
    )?;
    watcher.watch(Path::new(path_to_watch), RecursiveMode::Recursive)?;

    // 4. Watcher Event Loop
    while let Some(res) = rx.recv().await {
        match res {
            Ok(event) => {
                if event.kind.is_create() || event.kind.is_modify() {
                    for path in event.paths {
                        if path.is_file() {
                            sleep(Duration::from_millis(200));
                            process_file(&path, &client, &qdrant_url, &qdrant_api_key, &jina_api_key).await;
                        }
                    }
                }
            }
            Err(e) => println!("⚠️ Watcher error: {:?}", e),
        }
    }
    Ok(())
}

async fn process_file(
    file_path: &Path,
    client: &reqwest::Client,
    qdrant_url: &str,
    qdrant_api_key: &str,
    jina_api_key: &str,
) {
    if file_path.to_str().map_or(false, |s| s.contains(".DS_Store")) {
        return;
    }

    println!("📄 Processing file: {:?}", file_path);

    let content = match file_path.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()).as_deref() {
        Some("txt") | Some("csv") | Some("md") | Some("json") => {
            fs::read_to_string(file_path).ok()
        }
        Some("pdf") => {
            match pdf_extract::extract_text(file_path) {
                Ok(extracted) => {
                    let cleaned = extracted.trim().to_string();
                    if cleaned.is_empty() {
                        println!("⚠️ Warning: PDF extracted 0 characters from {:?}", file_path);
                        None
                    } else {
                        Some(cleaned)
                    }
                }
                Err(e) => {
                    println!("❌ Failed to parse PDF {:?}: Failure: {}", file_path, e);
                    None
                }
            }
        }
        _ => None,
    };

    if let Some(text) = content {
        let chunks = chunk_text(&text, 500);
        println!("🧩 Split into {} chunks. Fetching Jina AI embeddings...", chunks.len());

        let res = client
            .post("https://api.jina.ai/v1/embeddings")
            .header("Authorization", format!("Bearer {}", jina_api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({ 
                "model": "jina-embeddings-v2-base-en",
                "input": chunks 
            }))
            .send()
            .await;

        match res {
            Ok(response) => {
                if let Ok(json) = response.json::<serde_json::Value>().await {
                    if let Some(data) = json["data"].as_array() {
                        let keywords = extract_technical_terms(&text);
                        let source_name = file_path
                            .file_name()
                            .unwrap_or_default()
                            .to_str()
                            .unwrap_or("unknown");

                        let mut points = Vec::new();
                        for (i, item) in data.iter().enumerate() {
                            if let Some(vector) = item["embedding"].as_array() {
                                points.push(serde_json::json!({
                                    "id": Uuid::new_v4().to_string(),
                                    "vector": vector,
                                    "payload": {
                                        "text": chunks[i].clone(),
                                        "source": source_name,
                                        "keywords": keywords
                                    }
                                }));
                            }
                        }

                        let mut qdrant_req = client
                            .put(format!("{}/collections/hydra_docs/points", qdrant_url));
                        if !qdrant_api_key.is_empty() {
                            qdrant_req = qdrant_req.header("api-key", qdrant_api_key);
                        }

                        let qdrant_res = qdrant_req
                            .json(&serde_json::json!({ "points": points }))
                            .send()
                            .await;

                        if qdrant_res.is_ok() {
                            println!(
                                "💾 Qdrant: Successfully saved {} chunks from source '{}'",
                                points.len(),
                                source_name
                            );
                        } else {
                            println!("❌ Error saving to Qdrant.");
                        }
                    }
                }
            }
            Err(e) => println!("❌ Error calling Jina AI API: {:?}", e),
        }
    }
}

fn chunk_text(text: &str, size: usize) -> Vec<String> {
    text.chars()
        .collect::<Vec<char>>()
        .chunks(size)
        .map(|c| c.iter().collect())
        .collect()
}

fn extract_technical_terms(text: &str) -> Vec<String> {
    let keywords = vec!["LSTM-CNN", "YOLO", "Python", "TensorFlow", "Quantum", "Go", "Rust"];
    keywords
        .into_iter()
        .filter(|&term| text.contains(term))
        .map(|s| s.to_string())
        .collect()
}