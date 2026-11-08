package types

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetEffectiveCrossLanguageBridge(t *testing.T) {
	t.Parallel()

	t.Run("nil config and disabled block are off", func(t *testing.T) {
		t.Parallel()
		var nilCfg *RetrievalConfig
		assert.Nil(t, nilCfg.GetEffectiveCrossLanguageBridge())
		assert.Nil(t, (&RetrievalConfig{}).GetEffectiveCrossLanguageBridge())
		assert.Nil(t, (&RetrievalConfig{
			CrossLanguageBridge: &CrossLanguageBridgeConfig{Enabled: false, Mode: "concat"},
		}).GetEffectiveCrossLanguageBridge())
	})

	t.Run("mode normalizes to replace unless concat", func(t *testing.T) {
		t.Parallel()
		cfg := &RetrievalConfig{CrossLanguageBridge: &CrossLanguageBridgeConfig{
			Enabled: true,
			Mode:    "concat",
		}}
		assert.Equal(t, "concat", cfg.GetEffectiveCrossLanguageBridge().Mode)

		cfg.CrossLanguageBridge.Mode = ""
		assert.Equal(t, "replace", cfg.GetEffectiveCrossLanguageBridge().Mode)
		cfg.CrossLanguageBridge.Mode = "transmogrify"
		assert.Equal(t, "replace", cfg.GetEffectiveCrossLanguageBridge().Mode)
	})

	t.Run("returned copy never aliases the stored block", func(t *testing.T) {
		t.Parallel()
		stored := &CrossLanguageBridgeConfig{Enabled: true, Mode: "concat", ModelID: "m1"}
		cfg := &RetrievalConfig{CrossLanguageBridge: stored}
		effective := cfg.GetEffectiveCrossLanguageBridge()
		effective.ModelID = "mutated"
		assert.Equal(t, "m1", stored.ModelID)
	})
}

func TestGetEffectiveTestFilePenalty(t *testing.T) {
	t.Parallel()

	t.Run("nil config and disabled block are off", func(t *testing.T) {
		t.Parallel()
		var nilCfg *RetrievalConfig
		assert.Nil(t, nilCfg.GetEffectiveTestFilePenalty())
		assert.Nil(t, (&RetrievalConfig{}).GetEffectiveTestFilePenalty())
		assert.Nil(t, (&RetrievalConfig{
			TestFilePenalty: &TestFilePenaltyConfig{Enabled: false, Penalty: 0.5},
		}).GetEffectiveTestFilePenalty())
	})

	t.Run("out-of-range penalty falls back to default", func(t *testing.T) {
		t.Parallel()
		for _, invalid := range []float64{0, -0.5, 1.5} {
			cfg := &RetrievalConfig{TestFilePenalty: &TestFilePenaltyConfig{
				Enabled: true, Penalty: invalid,
			}}
			assert.Equal(t, DefaultTestFilePenalty,
				cfg.GetEffectiveTestFilePenalty().Penalty, "penalty %v", invalid)
		}
		cfg := &RetrievalConfig{TestFilePenalty: &TestFilePenaltyConfig{
			Enabled: true, Penalty: 1,
		}}
		// 1 is the valid no-op boundary: kept, and applyTestFilePenalty
		// treats it as a no-op.
		assert.InDelta(t, 1.0, cfg.GetEffectiveTestFilePenalty().Penalty, 1e-9)
	})

	t.Run("returned copy never aliases the stored block", func(t *testing.T) {
		t.Parallel()
		stored := &TestFilePenaltyConfig{Enabled: true, Penalty: 0.5}
		cfg := &RetrievalConfig{TestFilePenalty: stored}
		effective := cfg.GetEffectiveTestFilePenalty()
		effective.Penalty = 0.9
		assert.Equal(t, 0.5, stored.Penalty)
	})
}

// TestRetrievalConfigPortedBlocksJSONRoundTrip pins the JSON field names of
// the ported blocks: the settings API and the frontend share these keys, and
// a rename would silently orphan every stored configuration.
func TestRetrievalConfigPortedBlocksJSONRoundTrip(t *testing.T) {
	t.Parallel()

	in := `{
	  "embedding_top_k": 50,
	  "cross_language_bridge": {"enabled": true, "kb_ids": ["kb-1"], "mode": "concat", "model_id": "m"},
	  "test_file_penalty": {"enabled": true, "kb_ids": ["kb-1"], "penalty": 0.5}
	}`
	var cfg RetrievalConfig
	require.NoError(t, json.Unmarshal([]byte(in), &cfg))
	require.NotNil(t, cfg.CrossLanguageBridge)
	assert.True(t, cfg.CrossLanguageBridge.Enabled)
	assert.Equal(t, []string{"kb-1"}, cfg.CrossLanguageBridge.KBIDs)
	assert.Equal(t, "concat", cfg.CrossLanguageBridge.Mode)
	assert.Equal(t, "m", cfg.CrossLanguageBridge.ModelID)
	require.NotNil(t, cfg.TestFilePenalty)
	assert.True(t, cfg.TestFilePenalty.Enabled)
	assert.InDelta(t, 0.5, cfg.TestFilePenalty.Penalty, 1e-9)

	out, err := json.Marshal(&cfg)
	require.NoError(t, err)
	assert.Contains(t, string(out), `"cross_language_bridge"`)
	assert.Contains(t, string(out), `"test_file_penalty"`)

	// Absent blocks stay nil so the update handler can distinguish "leave
	// as configured" from "explicitly disabled".
	var bare RetrievalConfig
	require.NoError(t, json.Unmarshal([]byte(`{"embedding_top_k": 10}`), &bare))
	assert.Nil(t, bare.CrossLanguageBridge)
	assert.Nil(t, bare.TestFilePenalty)
}
