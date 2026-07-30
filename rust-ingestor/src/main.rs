use notify::{Watcher, RecursiveMode, Config, Result};
use std::path::Path;
use std::fs;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    let path_to_watch = "../data";
    println!("🚀 Hydra-Ingestor: Watching folder: {}", path_to_watch);

    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::RecommendedWatcher::new(tx, Config::default())?;
    watcher.watch(Path::new(path_to_watch), RecursiveMode::Recursive)?;

    let client = reqwest::Client::new();

    // --- NEW: Initialize Qdrant Database Collection ---
    let qdrant_init = serde_json::json!({
        "vectors": { "size": 384, "distance": "Cosine" }
    });
    let _ = client.put("http://localhost:6333/collections/hydra_docs")
        .json(&qdrant_init)
        .send().await;
    println!("🗄️ Database: 'hydra_docs' collection is ready.");

    // Start watching for files
    for res in rx {
        match res {
            Ok(event) => {
                if event.kind.is_create() {
                    let file_path = &event.paths[0];
                    if file_path.to_str().unwrap().contains(".DS_Store") { continue; }

                    println!("📄 New file detected: {:?}", file_path);
                    
                    let content = match file_path.extension().and_then(|s| s.to_str()) {
                        Some("txt") | Some("csv") | Some("md") | Some("json") => fs::read_to_string(file_path).ok(),
                        Some("pdf") => pdf_extract::extract_text(file_path).ok(),
                        _ => None,
                    };

                    if let Some(text) = content {
                        let chunks = chunk_text(&text, 500);
                        println!("🧩 Split into {} chunks. Sending to AI Service...", chunks.len());

                        // 1. Send to Python Inference
                        let res = client.post("http://localhost:8000/embed")
                            .json(&serde_json::json!({ "chunks": chunks }))
                            .send()
                            .await;

                        match res {
                            Ok(response) => {
                                if let Ok(json) = response.json::<serde_json::Value>().await {
                                    if let Some(vectors) = json["vectors"].as_array() {
                                        println!("🧠 AI Service: Generated {} vectors.", vectors.len());
                                        
                                        // --- NEW: Save to Qdrant ---
                                        let mut points = Vec::new();
                                        for (i, vector) in vectors.iter().enumerate() {
                                            let keywords = extract_technical_terms(&text);
                                            println!("🏷️ Tags extracted: {:?}", keywords);
                                            points.push(serde_json::json!({
                                                "id": Uuid::new_v4().to_string(),
                                                "vector": vector,
                                                "payload": {
                                                    "text": chunks[i].clone(),
                                                    "source": file_path.file_name().unwrap().to_str().unwrap(),
                                                    "keywords": keywords
                                                }
                                            }));
                                        }

                                        let qdrant_res = client.put("http://localhost:6333/collections/hydra_docs/points")
                                            .json(&serde_json::json!({ "points": points }))
                                            .send()
                                            .await;

                                        if qdrant_res.is_ok() {
                                            println!("💾 Qdrant: Successfully saved {} chunks to the database!", points.len());
                                        } else {
                                            println!("❌ Error saving to Qdrant.");
                                        }
                                    }
                                }
                            }
                            Err(_) => println!("❌ Error: Python Inference Service is not reachable."),
                        }
                    }
                }
            }
            Err(e) => println!("⚠️ Watcher error: {:?}", e),
        }
    }
    Ok(())
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
    keywords.into_iter()
        .filter(|&term| text.contains(term))
        .map(|s| s.to_string())
        .collect()
}