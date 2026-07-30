package main

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/go-resty/resty/v2"
	"github.com/gofiber/fiber/v2"
)

type QueryRequest struct {
	Query string `json:"query"`
}

// Thread-safe memory cache for tracking conversation history
type ChatMessage struct {
	Role    string
	Content string
}

var (
	chatHistory []ChatMessage
	mu          sync.Mutex 
)

func main() {
	app := fiber.New()

	app.Static("/", "./public")
	
	// Local resty client configuration
	client := resty.New().
		SetTimeout(60 * time.Second) // Give your M4 enough time to process deep queries if needed

	// --- Raw Vector Search Endpoint ---
	app.Post("/search", func(c *fiber.Ctx) error {
		req := new(QueryRequest)
		if err := c.BodyParser(req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "Invalid input"})
		}
		matches, err := fetchContext(client, req.Query)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(fiber.Map{"status": "success", "matches": matches})
	})

	// --- Local LLM Driver Synthesis Endpoint ---
	app.Post("/ask", func(c *fiber.Ctx) error {
		req := new(QueryRequest)
		if err := c.BodyParser(req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "Invalid input"})
		}

		// 1. Fetch relevant vector context blocks from Qdrant
		matches, err := fetchContext(client, req.Query)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"error": "Context collection failed"})
		}

		contextText := ""
		for i, match := range matches {
			payload, _ := match.(map[string]interface{})["payload"].(map[string]interface{})
			text, _ := payload["text"].(string)
			source, _ := payload["source"].(string)
			contextText += fmt.Sprintf("[%d] Source (%s): %s\n\n", i+1, source, text)
		}
		log.Println("🔍 DEBUG - Context fetched from Qdrant:")
		if contextText == "" {
			log.Println("⚠️ CONTEXT IS EMPTY! Qdrant found nothing.")
		} else {
			log.Println(contextText)
		}

		// 2. Format the local conversation history window (Last 4 messages)
		mu.Lock()
		historyText := ""
		if len(chatHistory) > 0 {
			historyText = "PREVIOUS CONVERSATION HISTORY:\n"
			start := 0
			if len(chatHistory) > 4 {
				start = len(chatHistory) - 4
			}
			for _, msg := range chatHistory[start:] {
				historyText += msg.Role + ": " + msg.Content + "\n"
			}
			historyText += "\n"
		}
		mu.Unlock()

		// 3. Construct the local Truth-Filter System Prompt
		systemPrompt := "You are the Hydra-RAG Generation Driver. You answer user questions using ONLY the provided text context.\n" +
			"CRITICAL RULES:\n" +
			"1. If the context is empty and the question cannot be answered using the PREVIOUS CONVERSATION HISTORY, reply exactly: 'I cannot find credible data.'\n" +
			"2. You may use the PREVIOUS CONVERSATION HISTORY to understand pronouns (like 'it' or 'they') or follow-up questions.\n" +
			"3. Cite the source names when answering.\n\n" +
			historyText +
			"CONTEXT FROM DATABASE:\n" + contextText + "\nUSER QUESTION: " + req.Query

		// 4. Fire payload directly to your local Ollama instance
		resp, err := client.R().
			SetHeader("Content-Type", "application/json").
			SetBody(map[string]interface{}{
				"model":  "llama3",
				"system": systemPrompt,
				"prompt": "CONTEXT:\n" + contextText + "\n\nUSER QUESTION: " + req.Query,
				"stream": false,
				"options": map[string]interface{}{
					"temperature": 0.0,
				},
			}).
			Post("http://localhost:11434/api/generate")

		if err != nil || resp.IsError() {
			log.Printf("🚨 Local Driver Error: %v", err)
			return c.Status(500).JSON(fiber.Map{"error": "Local LLM core unreachable"})
		}

		// 5. Parse clean, single-tier response from Ollama
		var ollamaRes map[string]interface{}
		if err := json.Unmarshal(resp.Body(), &ollamaRes); err != nil {
			return c.Status(500).JSON(fiber.Map{"error": "Failed to parse local intelligence output"})
		}
		
		finalAnswer, ok := ollamaRes["response"].(string)
		if !ok {
			finalAnswer = "Local model failed to yield string output response structural format."
		}

		// 6. Append fresh interactions to the localized historical map
		mu.Lock()
		chatHistory = append(chatHistory, ChatMessage{Role: "User", Content: req.Query})
		chatHistory = append(chatHistory, ChatMessage{Role: "AI", Content: finalAnswer})
		mu.Unlock()

		return c.JSON(fiber.Map{
			"status":   "success",
			"question": req.Query,
			"answer":   finalAnswer,
		})
	})

	log.Println("🏎️  Hydra Gateway with Local Memory running on http://localhost:3000")
	log.Fatal(app.Listen(":3000"))
}

func fetchContext(client *resty.Client, query string) ([]interface{}, error) {
	var aiResponse map[string]interface{}
	_, err := client.R().
		SetBody(map[string]interface{}{"chunks": []string{query}}).
		SetResult(&aiResponse).
		Post("http://localhost:8000/embed")
	if err != nil {
		return nil, fmt.Errorf("AI embedding failed")
	}

	vectors, _ := aiResponse["vectors"].([]interface{})
	var qdrantResponse map[string]interface{}
    _, err = client.R().
        SetBody(map[string]interface{}{
            "search_params": map[string]interface{}{"hnsw_ef": 128},
            "vector":       vectors[0],
            "limit":        5,
            "with_payload": true,
            // Hybrid logic: The AI can now force-match technical keywords
            "filter": map[string]interface{}{
                "should": []map[string]interface{}{
                    {"key": "keywords", "match": map[string]interface{}{"any": []string{"LSTM-CNN", "YOLO"}}},
                },
            },
        }).
        SetResult(&qdrantResponse).
        Post("http://localhost:6333/collections/hydra_docs/points/search")
	if err != nil {
		return nil, fmt.Errorf("Database fetch failed")
	}

	result, _ := qdrantResponse["result"].([]interface{})
	return result, nil
}