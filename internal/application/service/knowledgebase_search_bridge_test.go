package service

import (
	"context"
	"testing"

	"github.com/Tencent/WeKnora/internal/models/chat"
	"github.com/Tencent/WeKnora/internal/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// bridgeChatModel records the translation request and returns a canned reply.
type bridgeChatModel struct {
	replies  []string // popped per call; last one repeats
	err      error
	lastMsgs []chat.Message
	lastOpts *chat.ChatOptions
	modelID  string
}

func (m *bridgeChatModel) Chat(_ context.Context, messages []chat.Message,
	opts *chat.ChatOptions,
) (*types.ChatResponse, error) {
	m.lastMsgs = messages
	m.lastOpts = opts
	if m.err != nil {
		return nil, m.err
	}
	reply := ""
	if len(m.replies) > 0 {
		reply = m.replies[0]
		m.replies = m.replies[1:]
	}
	return &types.ChatResponse{Content: reply}, nil
}

func (*bridgeChatModel) ChatStream(context.Context, []chat.Message, *chat.ChatOptions) (<-chan types.StreamResponse, error) {
	return nil, nil
}

func (m *bridgeChatModel) GetModelName() string { return "bridge-fake" }
func (m *bridgeChatModel) GetModelID() string   { return m.modelID }

// newBridgeService builds a service whose default chat model is a fresh
// recording fake, so parallel subtests never share model state.
func newBridgeService(model *bridgeChatModel) *knowledgeBaseService {
	return &knowledgeBaseService{
		modelService: &stubModelService{
			chatModel: model,
			availableModels: []*types.Model{
				{ID: "chat-1", Type: types.ModelTypeKnowledgeQA, Status: types.ModelStatusActive},
			},
		},
	}
}

func TestCJKDominantQuery(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		q    string
		want bool
	}{
		{"chinese question", "日期格式化是在哪个类里实现的", true},
		{"japanese question", "シリアライズの設定はどこでやる", true},
		{"english question", "where is date formatting implemented", false},
		{"identifier with one CJK particle", "TypeAdapter の使い方 a", false},
		{"short CJK only", "在哪里", false},
		{"mixed but CJK dominant", "JSONのデシリアライズに失敗した時の挙動は", true},
		{"empty", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.want, cjkDominantQuery(tc.q))
		})
	}
}

func TestApplyCrossLanguageBridge(t *testing.T) {
	t.Parallel()

	kbs := []*types.KnowledgeBase{{ID: "kb-code"}}

	newParams := func() types.SearchParams {
		return types.SearchParams{
			QueryText:      "日期格式化是在哪个类里实现的",
			QueryEmbedding: []float32{0.1, 0.2},
		}
	}

	t.Run("replaces the query and drops the stale embedding", func(t *testing.T) {
		t.Parallel()
		chatModel := &bridgeChatModel{
			replies: []string{"Where is date formatting implemented?"},
		}
		s := newBridgeService(chatModel)
		params := newParams()
		s.applyCrossLanguageBridge(context.Background(), kbs, &params,
			&types.CrossLanguageBridgeConfig{Enabled: true})

		assert.Equal(t, "Where is date formatting implemented?", params.QueryText)
		assert.Empty(t, params.QueryEmbedding)

		require.Len(t, chatModel.lastMsgs, 2)
		assert.Equal(t, "system", chatModel.lastMsgs[0].Role)
		assert.Equal(t, bridgeTranslateSystemPrompt, chatModel.lastMsgs[0].Content)
		assert.Equal(t, "日期格式化是在哪个类里实现的", chatModel.lastMsgs[1].Content)
		require.NotNil(t, chatModel.lastOpts)
		assert.InDelta(t, 0.0, chatModel.lastOpts.Temperature, 1e-9)
		assert.Equal(t, crossLanguageBridgeMaxTokens, chatModel.lastOpts.MaxCompletionTokens)
		require.NotNil(t, chatModel.lastOpts.Thinking)
		assert.False(t, *chatModel.lastOpts.Thinking)
	})

	t.Run("concat mode keeps the original query", func(t *testing.T) {
		t.Parallel()
		chatModel := &bridgeChatModel{
			replies: []string{"Where is date formatting implemented?"},
		}
		s := newBridgeService(chatModel)
		params := newParams()
		s.applyCrossLanguageBridge(context.Background(), kbs, &params,
			&types.CrossLanguageBridgeConfig{Enabled: true, Mode: "concat"})

		assert.Equal(t,
			"日期格式化是在哪个类里实现的 Where is date formatting implemented?",
			params.QueryText)
	})

	t.Run("non-CJK query passes through without a model call", func(t *testing.T) {
		t.Parallel()
		chatModel := &bridgeChatModel{}
		s := newBridgeService(chatModel)
		params := types.SearchParams{QueryText: "where is TypeAdapter used"}
		s.applyCrossLanguageBridge(context.Background(), kbs, &params,
			&types.CrossLanguageBridgeConfig{Enabled: true})

		assert.Equal(t, "where is TypeAdapter used", params.QueryText)
		assert.Nil(t, chatModel.lastMsgs)
	})

	t.Run("kb outside the scope is untouched", func(t *testing.T) {
		t.Parallel()
		chatModel := &bridgeChatModel{}
		s := newBridgeService(chatModel)
		params := newParams()
		s.applyCrossLanguageBridge(context.Background(), kbs, &params,
			&types.CrossLanguageBridgeConfig{Enabled: true, KBIDs: []string{"kb-other"}})

		assert.Equal(t, "日期格式化是在哪个类里实现的", params.QueryText)
		assert.NotNil(t, params.QueryEmbedding)
		assert.Nil(t, chatModel.lastMsgs)
	})

	t.Run("model failure keeps the original query", func(t *testing.T) {
		t.Parallel()
		chatModel := &bridgeChatModel{err: context.DeadlineExceeded}
		s := newBridgeService(chatModel)
		params := newParams()
		s.applyCrossLanguageBridge(context.Background(), kbs, &params,
			&types.CrossLanguageBridgeConfig{Enabled: true})

		assert.Equal(t, "日期格式化是在哪个类里实现的", params.QueryText)
		assert.NotNil(t, params.QueryEmbedding)
	})

	t.Run("empty translation keeps the original query", func(t *testing.T) {
		t.Parallel()
		chatModel := &bridgeChatModel{replies: []string{"   "}}
		s := newBridgeService(chatModel)
		params := newParams()
		s.applyCrossLanguageBridge(context.Background(), kbs, &params,
			&types.CrossLanguageBridgeConfig{Enabled: true})

		assert.Equal(t, "日期格式化是在哪个类里实现的", params.QueryText)
	})
}

func TestBridgeScopedKB(t *testing.T) {
	t.Parallel()

	kbs := []*types.KnowledgeBase{{ID: "kb-a"}, {ID: "kb-b"}}
	assert.True(t, bridgeScopedKB(kbs,
		&types.CrossLanguageBridgeConfig{Enabled: true}))
	assert.True(t, bridgeScopedKB(kbs,
		&types.CrossLanguageBridgeConfig{Enabled: true, KBIDs: []string{"kb-b"}}))
	assert.False(t, bridgeScopedKB(kbs,
		&types.CrossLanguageBridgeConfig{Enabled: true, KBIDs: []string{"kb-c"}}))
	assert.False(t, bridgeScopedKB(nil,
		&types.CrossLanguageBridgeConfig{Enabled: true, KBIDs: []string{"kb-a"}}))
}
