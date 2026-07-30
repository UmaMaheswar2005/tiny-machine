from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import numpy as np

app = FastAPI()

# Load the model (In a production engine, we'd use raw ONNX, 
# but sentence-transformers handles the ONNX logic well for now)
model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingRequest(BaseModel):
    chunks: list[str]

@app.post("/embed")
async def embed_chunks(request: EmbeddingRequest):
    try:
        # Turn text into math
        embeddings = model.encode(request.chunks)
        # Convert numpy arrays to lists for JSON transport
        return {"vectors": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)