package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/go-resty/resty/v2"
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type QueryRequest struct {
	Query     string `json:"query"`
	SessionID string `json:"session_id"` // Added SessionID field
}

// Thread-safe memory cache for tracking conversation history per session
type ChatMessage struct {
	Role    string `json:"role"`    // "user" or "assistant"
	Content string `json:"content"`
}

var (
	// Map session_id -> list of chat messages
	sessionStore = make(map[string][]ChatMessage)
	mu           sync.RWMutex
)

func main() {
	app := fiber.New()

	app.Static("/", "./public")

	// Local resty client configuration
	client := resty.New().
		SetTimeout(60 * time.Second)

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

	// --- 1. Get All Active Session IDs (For Sidebar List) ---
	app.Get("/sessions", func(c *fiber.Ctx) error {
		mu.RLock()
		defer mu.RUnlock()

		sessions := make([]string, 0, len(sessionStore))
		for id := range sessionStore {
			sessions = append(sessions, id)
		}
		return c.JSON(fiber.Map{"sessions": sessions})
	})

	// --- 2. Get Messages for a Specific Session (Load Old Chat) ---
	app.Get("/sessions/:id", func(c *fiber.Ctx) error {
		sessionID := c.Params("id")

		mu.RLock()
		history, exists := sessionStore[sessionID]
		mu.RUnlock()

		if !exists {
			return c.Status(404).JSON(fiber.Map{"error": "Session not found"})
		}
		return c.JSON(fiber.Map{"session_id": sessionID, "messages": history})
	})

	// --- 3. Delete a Session ---
	app.Delete("/sessions/:id", func(c *fiber.Ctx) error {
		sessionID := c.Params("id")

		mu.Lock()
		delete(sessionStore, sessionID)
		mu.Unlock()

		return c.JSON(fiber.Map{"status": "deleted", "session_id": sessionID})
	})

	// --- LLM Driver Synthesis Endpoint ---
	app.Post("/ask", func(c *fiber.Ctx) error {
		req := new(QueryRequest)
		if err := c.BodyParser(req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "Invalid input"})
		}

		// Ensure session_id exists (generate UUID if missing)
		if req.SessionID == "" {
			req.SessionID = uuid.New().String()
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

		systemPrompt := `You are Hydra-RAG, an elite, highly intelligent, and analytical AI assistant.

            ### CORE DIRECTIVES:

            1. TASK-BASED GENERATION MODE (General Prompts)
            - TRIGGER: The user asks you to perform a general task (e.g., "write a python script", "explain REST APIs", "hello").
            - RULE: Ignore [RETRIEVED CONTEXT] unless the user explicitly asks to base the task on local files. Generate the response directly using your AI capabilities. Do NOT state "I didn't find this in your local data" for simple generation requests.

            2. LOCAL DATA MODE (Strict & Precise)
            - TRIGGER: The user explicitly asks about local documents, codebases, or stored portfolio data, AND [RETRIEVED CONTEXT] contains relevant information.
            - RULE: Answer STRICTLY based on [RETRIEVED CONTEXT]. Do not hallucinate external details.

            3. GLOBAL KNOWLEDGE MODE
            - TRIGGER: The user asks about local/internal topic, but [RETRIEVED CONTEXT] is empty or irrelevant.
            - RULE: Briefly state: *"No matching local context found, but here is a breakdown based on general knowledge:"* and provide the answer.`

		// 2. Fetch conversation history for this session (Last 6 turns)
		mu.RLock()
		history := sessionStore[req.SessionID]
		mu.RUnlock()

		// 3. Build OpenAI-compatible messages array with history
		messagesPayload := []map[string]string{
			{"role": "system", "content": systemPrompt},
		}

		// Append last 6 turns from history to the payload
		start := 0
		if len(history) > 6 {
			start = len(history) - 6
		}
		for _, msg := range history[start:] {
			messagesPayload = append(messagesPayload, map[string]string{
				"role":    msg.Role,
				"content": msg.Content,
			})
		}

		// Append the current query + context block as the latest user message
		userContent := fmt.Sprintf("[RETRIEVED CONTEXT]:\n%s\n\nUSER QUESTION: %s", contextText, req.Query)
		messagesPayload = append(messagesPayload, map[string]string{
			"role":    "user",
			"content": userContent,
		})

		// 4. Resolve external LLM configuration
		apiKey := os.Getenv("LLM_API_KEY")
		apiURL := os.Getenv("LLM_BASE_URL")
		if apiURL == "" {
			apiURL = "https://api.groq.com/openai/v1/chat/completions"
		}
		modelName := os.Getenv("LLM_MODEL")
		if modelName == "" {
			modelName = "llama-3.1-8b-instant"
		}

		// 5. Send request using OpenAI-compatible chat format
		resp, err := client.R().
			SetHeader("Content-Type", "application/json").
			SetHeader("Authorization", "Bearer "+apiKey).
			SetBody(map[string]interface{}{
				"model":       modelName,
				"messages":    messagesPayload,
				"temperature": 0.0,
			}).
			Post(apiURL)

		if err != nil || resp.IsError() {
			log.Printf("🚨 External Driver Error: %v | Response: %s", err, resp.String())
			return c.Status(500).JSON(fiber.Map{"error": "External LLM API unreachable"})
		}

		// 6. Parse OpenAI choices format response
		var apiRes map[string]interface{}
		if err := json.Unmarshal(resp.Body(), &apiRes); err != nil {
			return c.Status(500).JSON(fiber.Map{"error": "Failed to parse intelligence output"})
		}

		finalAnswer := "Model failed to yield a response structure."
		if choices, ok := apiRes["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if message, ok := choice["message"].(map[string]interface{}); ok {
					if content, ok := message["content"].(string); ok {
						finalAnswer = content
					}
				}
			}
		}

		// 7. Append fresh interaction to session history
		mu.Lock()
		sessionStore[req.SessionID] = append(sessionStore[req.SessionID],
			ChatMessage{Role: "user", Content: req.Query},
			ChatMessage{Role: "assistant", Content: finalAnswer},
		)
		mu.Unlock()

		return c.JSON(fiber.Map{
			"status":     "success",
			"session_id": req.SessionID,
			"question":   req.Query,
			"answer":     finalAnswer,
		})
	})

	log.Println("🏎️ Hydra Gateway with Session Memory running on http://localhost:3000")
	log.Fatal(app.Listen(":3000"))
}

func fetchContext(client *resty.Client, query string) ([]interface{}, error) {
	// Dynamically resolve environment variables for Docker network compatibility
	inferenceURL := os.Getenv("INFERENCE_URL")
	if inferenceURL == "" {
		inferenceURL = "http://localhost:8000"
	}

	qdrantURL := os.Getenv("QDRANT_URL")
	if qdrantURL == "" {
		qdrantURL = "http://localhost:6333"
	}

	// 1. Send query to inference service for embedding
	var aiResponse map[string]interface{}
	_, err := client.R().
		SetBody(map[string]interface{}{"chunks": []string{query}}).
		SetResult(&aiResponse).
		Post(fmt.Sprintf("%s/embed", inferenceURL))
	if err != nil {
		return nil, fmt.Errorf("AI embedding failed: %v", err)
	}

	vectors, _ := aiResponse["vectors"].([]interface{})
	if len(vectors) == 0 {
		return nil, fmt.Errorf("No vectors returned from inference service")
	}

	// 2. Query Qdrant vector database
	var qdrantResponse map[string]interface{}
	_, err = client.R().
		SetBody(map[string]interface{}{
			"search_params": map[string]interface{}{"hnsw_ef": 128},
			"vector":        vectors[0],
			"limit":         5,
			"with_payload":  true,
			"filter": map[string]interface{}{
				"should": []map[string]interface{}{
					{"key": "keywords", "match": map[string]interface{}{"any": []string{"LSTM-CNN", "YOLO"}}},
				},
			},
		}).
		SetResult(&qdrantResponse).
		Post(fmt.Sprintf("%s/collections/hydra_docs/points/search", qdrantURL))
	if err != nil {
		return nil, fmt.Errorf("Database fetch failed: %v", err)
	}

	result, _ := qdrantResponse["result"].([]interface{})
	return result, nil
}