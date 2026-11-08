package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Tencent/WeKnora/internal/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func putRetrievalConfig(t *testing.T, existing *types.RetrievalConfig, body string) (
	*types.RetrievalConfig, int, string,
) {
	t.Helper()
	tenants := &defaultSettingsTenantService{}
	h := &TenantHandler{service: tenants}
	ctx := context.WithValue(context.Background(), types.TenantIDContextKey, uint64(7))
	ctx = context.WithValue(ctx, types.TenantInfoContextKey,
		&types.Tenant{ID: 7, RetrievalConfig: existing})
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPut, "/tenants/kv/retrieval-config",
		bytes.NewReader([]byte(body)))
	c.Request = c.Request.WithContext(ctx)
	c.Request.Header.Set("Content-Type", "application/json")
	h.updateTenantRetrievalConfigInternal(c)

	if len(c.Errors) > 0 {
		return nil, w.Code, c.Errors.Last().Error()
	}
	require.NotNil(t, tenants.saved)
	require.NotNil(t, tenants.saved.RetrievalConfig)
	return tenants.saved.RetrievalConfig, w.Code, ""
}

// TestUpdateRetrievalConfigPreservesPortedBlocks pins the merge rule for the
// cross_language_bridge and test_file_penalty blocks: the settings UI form
// submits only the fields it renders, so an update that omits a block must
// not delete it.
func TestUpdateRetrievalConfigPreservesPortedBlocks(t *testing.T) {
	t.Parallel()

	existing := &types.RetrievalConfig{
		CrossLanguageBridge: &types.CrossLanguageBridgeConfig{
			Enabled: true, KBIDs: []string{"kb-code"}, Mode: "replace", ModelID: "m1",
		},
		TestFilePenalty: &types.TestFilePenaltyConfig{
			Enabled: true, KBIDs: []string{"kb-code"}, Penalty: 0.5,
		},
	}

	t.Run("form-shaped update keeps both blocks", func(t *testing.T) {
		t.Parallel()
		saved, code, errs := putRetrievalConfig(t, existing,
			`{"embedding_top_k": 20, "vector_threshold": 0.2}`)
		require.Empty(t, errs)
		assert.Equal(t, http.StatusOK, code)
		assert.Equal(t, existing.CrossLanguageBridge, saved.CrossLanguageBridge)
		assert.Equal(t, existing.TestFilePenalty, saved.TestFilePenalty)
		assert.Equal(t, 20, saved.EmbeddingTopK)
	})

	t.Run("explicit disabled block is honored", func(t *testing.T) {
		t.Parallel()
		saved, _, errs := putRetrievalConfig(t, existing,
			`{"cross_language_bridge": {"enabled": false}}`)
		require.Empty(t, errs)
		require.NotNil(t, saved.CrossLanguageBridge)
		assert.False(t, saved.CrossLanguageBridge.Enabled)
		// The untouched sibling block survives.
		assert.Equal(t, existing.TestFilePenalty, saved.TestFilePenalty)
	})

	t.Run("fresh config with absent blocks stores nothing", func(t *testing.T) {
		t.Parallel()
		saved, _, errs := putRetrievalConfig(t, nil, `{"embedding_top_k": 10}`)
		require.Empty(t, errs)
		assert.Nil(t, saved.CrossLanguageBridge)
		assert.Nil(t, saved.TestFilePenalty)
	})

	t.Run("invalid mode is rejected", func(t *testing.T) {
		t.Parallel()
		_, _, errs := putRetrievalConfig(t, nil,
			`{"cross_language_bridge": {"enabled": true, "mode": "sideways"}}`)
		require.NotEmpty(t, errs)
		assert.Contains(t, errs, "mode must be")
	})

	t.Run("invalid penalty is rejected", func(t *testing.T) {
		t.Parallel()
		for _, body := range []string{
			`{"test_file_penalty": {"enabled": true, "penalty": 0}}`,
			`{"test_file_penalty": {"enabled": true, "penalty": 1.5}}`,
			`{"test_file_penalty": {"enabled": true, "penalty": -0.5}}`,
		} {
			_, _, errs := putRetrievalConfig(t, nil, body)
			require.NotEmpty(t, errs, body)
			assert.Contains(t, errs, "penalty must be", body)
		}
	})

	t.Run("valid blocks round-trip", func(t *testing.T) {
		t.Parallel()
		saved, _, errs := putRetrievalConfig(t, nil, `{
			"cross_language_bridge": {"enabled": true, "kb_ids": ["kb-code"], "mode": "concat"},
			"test_file_penalty": {"enabled": true, "kb_ids": ["kb-code"], "penalty": 0.5}
		}`)
		require.Empty(t, errs)
		require.NotNil(t, saved.CrossLanguageBridge)
		assert.True(t, saved.CrossLanguageBridge.Enabled)
		assert.Equal(t, "concat", saved.CrossLanguageBridge.Mode)
		require.NotNil(t, saved.TestFilePenalty)
		assert.InDelta(t, 0.5, saved.TestFilePenalty.Penalty, 1e-9)
	})
}

// TestUpdateRetrievalConfigMergeDoesNotMutateRequestTenant guards against the
// merge rule aliasing the caller's config object into the saved tenant.
func TestUpdateRetrievalConfigMergeDoesNotMutateRequestTenant(t *testing.T) {
	t.Parallel()

	existing := &types.RetrievalConfig{
		CrossLanguageBridge: &types.CrossLanguageBridgeConfig{Enabled: true},
	}
	before, err := json.Marshal(existing.CrossLanguageBridge)
	require.NoError(t, err)

	saved, _, errs := putRetrievalConfig(t, existing, `{"embedding_top_k": 5}`)
	require.Empty(t, errs)
	require.NotNil(t, saved.CrossLanguageBridge)

	after, err := json.Marshal(existing.CrossLanguageBridge)
	require.NoError(t, err)
	assert.JSONEq(t, string(before), string(after))
}
