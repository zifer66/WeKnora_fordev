package types

import (
	"database/sql/driver"
	"encoding/json"
)

// RetrievalConfig holds the global retrieval/search configuration for a tenant.
// This replaces the retrieval-related fields previously scattered in ConversationConfig
// and ChatHistoryConfig. Both knowledge search and message search share these parameters.
//
// Stored as a JSONB column on the tenants table, managed via the settings UI
// at /tenants/kv/retrieval-config.
type RetrievalConfig struct {
	// EmbeddingTopK is the maximum number of chunks returned by vector search (default: 50)
	EmbeddingTopK int `json:"embedding_top_k"`
	// VectorThreshold is the minimum vector similarity score (0-1, default: 0.15)
	VectorThreshold float64 `json:"vector_threshold"`
	// KeywordThreshold is the minimum keyword match score (0-1, default: 0.3)
	KeywordThreshold float64 `json:"keyword_threshold"`
	// RerankTopK is the maximum number of results after reranking (default: 10)
	RerankTopK int `json:"rerank_top_k"`
	// RerankThreshold is the minimum rerank score (-10 to 10, default: 0.2)
	RerankThreshold float64 `json:"rerank_threshold"`
	// RerankModelID is the ID of the rerank model to use (required for search)
	RerankModelID string `json:"rerank_model_id"`

	// RRFK is the smoothing constant of Reciprocal Rank Fusion. Larger values
	// flatten the curve, reducing the bias towards top-1 results.
	// Default: 60. Sensible range: 30..100 depending on corpus size.
	RRFK int `json:"rrf_k,omitempty"`
	// RRFVectorWeight is the weight applied to the vector retriever inside RRF.
	// RRFVectorWeight + RRFKeywordWeight should usually sum to 1.0 but the math
	// works for any positive weights. Default: 0.7.
	RRFVectorWeight float64 `json:"rrf_vector_weight,omitempty"`
	// RRFKeywordWeight is the keyword counterpart. Default: 0.3.
	RRFKeywordWeight float64 `json:"rrf_keyword_weight,omitempty"`

	// CrossLanguageBridge translates a CJK-dominant query into English before
	// retrieval. It exists for knowledge bases whose corpus is Latin-script
	// (source code, English documents): such a question has no lexical bridge
	// into the corpus and both retrieval legs miss. Measured on a 20-question
	// Chinese/Japanese to English Java-repo golden set, retrieval top-1 rose
	// from 0/20 to 4/20 with the bridge alone, and to 12/20 stacked with
	// test-file down-weighting and chunk rerank, at zero agent cost.
	// A corpus written in Chinese/Japanese must keep it disabled: translation
	// would move its queries away from the language of its own chunks.
	// Disabled by default; managed via the retrieval-config KV API.
	CrossLanguageBridge *CrossLanguageBridgeConfig `json:"cross_language_bridge,omitempty"`
	// TestFilePenalty downscores chunks whose document title identifies a test
	// file, unless the query itself asks about tests. On code QA the lexically
	// similar test chunk regularly outranks the implementation the question
	// asks for (7 of 16 retrieval misses on the cross-lingual golden set had a
	// test file at top-1 with the gold one to five ranks below). Halving the
	// fused score recovered 2 top-1 and 4 top-5 hits without hurting sets
	// whose gold answers legitimately include test files, thanks to the
	// query gate. Disabled by default; managed via the retrieval-config KV API.
	TestFilePenalty *TestFilePenaltyConfig `json:"test_file_penalty,omitempty"`
}

// DefaultCrossLanguageBridgeMode searches only the translation.
const DefaultCrossLanguageBridgeMode = "replace"

// DefaultTestFilePenalty is the multiplier the research grid search picked:
// small enough to demote lexically similar test chunks below the
// implementation, large enough that a strong match can still surface.
const DefaultTestFilePenalty = 0.5

// CrossLanguageBridgeConfig scopes and parameterizes CJK to English query
// translation. Every field defaults to "off" or "workspace default", so an
// absent or zero-value block never changes retrieval behavior.
type CrossLanguageBridgeConfig struct {
	// Enabled turns the bridge on. Default false.
	Enabled bool `json:"enabled"`
	// KBIDs scopes the bridge to these knowledge bases; a search whose KBs
	// are all outside the list is untouched. Empty means every KB of the
	// workspace, which is only sensible when the whole workspace indexes
	// Latin-script corpora.
	KBIDs []string `json:"kb_ids,omitempty"`
	// Mode selects what is searched: "replace" (default) searches the
	// translation only, "concat" searches the original query joined with the
	// translation. Concat keeps original-language keywords in play at the
	// cost of noisier matches; the golden set preferred replace.
	Mode string `json:"mode,omitempty"`
	// ModelID is the chat model used for translation. Empty picks the
	// workspace default chat model.
	ModelID string `json:"model_id,omitempty"`
}

// TestFilePenaltyConfig scopes the test-file down-weighting.
type TestFilePenaltyConfig struct {
	// Enabled turns the down-weighting on. Default false.
	Enabled bool `json:"enabled"`
	// KBIDs scopes the penalty to chunks of these knowledge bases. Empty
	// means every KB of the workspace, which is only sensible when the
	// workspace only hosts code corpora.
	KBIDs []string `json:"kb_ids,omitempty"`
	// Penalty multiplies the fused score of test-file chunks. Valid range
	// (0, 1]; 1 would be a no-op. Default 0.5.
	Penalty float64 `json:"penalty"`
}

// DefaultRetrievalTopK is the retrieval depth used when a caller supplies no
// usable TopK / MatchCount. HybridSearch also floors its over-retrieval pool
// at this value, so the fallback and the pool it draws from stay in step by
// construction rather than by two independently maintained literals.
const DefaultRetrievalTopK = 50

// MaxRequestedResults caps the result counts a retrieval API caller may ask
// for (knowledge-search match_count, rerank.top_k). knowledge-search runs one
// search per scoped document or tag set and reranks every candidate, so an
// unbounded count multiplied into tens of thousands of billed rerank passages.
const MaxRequestedResults = 200

// GetEffectiveEmbeddingTopK returns EmbeddingTopK with a fallback default.
func (c *RetrievalConfig) GetEffectiveEmbeddingTopK() int {
	if c == nil || c.EmbeddingTopK <= 0 {
		return DefaultRetrievalTopK
	}
	return c.EmbeddingTopK
}

// GetEffectiveVectorThreshold returns VectorThreshold with a fallback default.
func (c *RetrievalConfig) GetEffectiveVectorThreshold() float64 {
	if c == nil || c.VectorThreshold <= 0 {
		return 0.15
	}
	return c.VectorThreshold
}

// GetEffectiveKeywordThreshold returns KeywordThreshold with a fallback default.
func (c *RetrievalConfig) GetEffectiveKeywordThreshold() float64 {
	if c == nil || c.KeywordThreshold <= 0 {
		return 0.3
	}
	return c.KeywordThreshold
}

// GetEffectiveRerankTopK returns RerankTopK with a fallback default.
func (c *RetrievalConfig) GetEffectiveRerankTopK() int {
	if c == nil || c.RerankTopK <= 0 {
		return 10
	}
	return c.RerankTopK
}

// GetEffectiveRerankThreshold returns RerankThreshold with a fallback default.
func (c *RetrievalConfig) GetEffectiveRerankThreshold() float64 {
	if c == nil {
		return 0.2
	}
	return c.RerankThreshold
}

// GetEffectiveRRFK returns the RRF smoothing constant with a fallback default.
func (c *RetrievalConfig) GetEffectiveRRFK() int {
	if c == nil || c.RRFK <= 0 {
		return 60
	}
	return c.RRFK
}

// GetEffectiveRRFWeights returns vector / keyword weights with sensible defaults.
// When neither weight is set explicitly, returns 0.7 / 0.3.
func (c *RetrievalConfig) GetEffectiveRRFWeights() (vector, keyword float64) {
	if c == nil || (c.RRFVectorWeight == 0 && c.RRFKeywordWeight == 0) {
		return 0.7, 0.3
	}
	v := c.RRFVectorWeight
	k := c.RRFKeywordWeight
	if v <= 0 {
		v = 0.7
	}
	if k <= 0 {
		k = 0.3
	}
	return v, k
}

// GetEffectiveCrossLanguageBridge returns the bridge config with defaults
// normalized, or nil when the bridge is off (config absent, disabled). The
// returned copy is safe to mutate: it never aliases the stored config.
func (c *RetrievalConfig) GetEffectiveCrossLanguageBridge() *CrossLanguageBridgeConfig {
	if c == nil || c.CrossLanguageBridge == nil || !c.CrossLanguageBridge.Enabled {
		return nil
	}
	out := *c.CrossLanguageBridge
	if out.Mode != "concat" {
		// replace is the default; an unknown mode falls back to it rather
		// than silently skipping the bridge.
		out.Mode = DefaultCrossLanguageBridgeMode
	}
	return &out
}

// GetEffectiveTestFilePenalty returns the test-file penalty config, or nil
// when the feature is off (config absent or disabled). A penalty outside
// (0, 1] — only possible for rows written without the handler validation —
// falls back to the default multiplier instead of trusting a bad value.
// The returned copy never aliases the stored config.
func (c *RetrievalConfig) GetEffectiveTestFilePenalty() *TestFilePenaltyConfig {
	if c == nil || c.TestFilePenalty == nil || !c.TestFilePenalty.Enabled {
		return nil
	}
	out := *c.TestFilePenalty
	if out.Penalty <= 0 || out.Penalty > 1 {
		out.Penalty = DefaultTestFilePenalty
	}
	return &out
}

// Value implements the driver.Valuer interface for database serialization
func (c RetrievalConfig) Value() (driver.Value, error) {
	return json.Marshal(c)
}

// Scan implements the sql.Scanner interface for database deserialization
func (c *RetrievalConfig) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	b, ok := value.([]byte)
	if !ok {
		return nil
	}
	return json.Unmarshal(b, c)
}
