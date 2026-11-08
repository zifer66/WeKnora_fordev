package service

import (
	"context"
	"regexp"
	"strings"
	"unicode"

	"github.com/Tencent/WeKnora/internal/logger"
	"github.com/Tencent/WeKnora/internal/models/chat"
	"github.com/Tencent/WeKnora/internal/types"
	secutils "github.com/Tencent/WeKnora/internal/utils"
)

// crossLanguageBridgeMaxTokens bounds the translation reply. One English
// question is all that is needed; the cap also keeps a model that starts to
// explain instead of translating from delaying the retrieval measurably.
const crossLanguageBridgeMaxTokens = 256

// bridgeTranslateSystemPrompt is the translation instruction. It keeps
// technical nouns on their conventional English names (the words the corpus
// actually uses) and preserves identifiers verbatim, because losing the one
// English identifier a CJK question carries was the failure mode of naive
// translation in early trials.
const bridgeTranslateSystemPrompt = "你是代码检索查询翻译器。把中文/日文问题翻译成英文，用于在英文语料（代码库/文档）中检索。" +
	"技术名词用常见英文译名（日期格式→date format、序列化→serialize、适配器→adapter、" +
	"枚举→enum、缩进/美化输出→indent/pretty print、严格程度→strictness 等）；" +
	"问题里已有的英文标识符（类名/方法名/注解/参数）原样保留。只输出一句英文问题，" +
	"不要解释、不要加引号。"

// cjkCharClass matches the CJK ideograph and kana ranges the dominance check
// counts. Compiled once; questions are short.
var cjkCharClass = regexp.MustCompile(`[\x{3040}-\x{30FF}\x{3400}-\x{4DBF}\x{4E00}-\x{9FFF}\x{F900}-\x{FAFF}\x{3005}]`)

// cjkDominantQuery reports whether the question is asked in Chinese or
// Japanese: at least four CJK characters and more CJK than Latin letters.
// Fewer than four CJK characters is usually a greeting particle around an
// otherwise Latin question, which retrieval already serves; translating
// those would only drop the exact terms. Identifier-carrying mixed questions
// ("TypeAdapter はどう使う") stay CJK-dominant here when they are, and the
// translation prompt preserves their identifiers.
func cjkDominantQuery(q string) bool {
	cjk := len(cjkCharClass.FindAllString(q, -1))
	if cjk < 4 {
		return false
	}
	latin := 0
	for _, r := range q {
		if unicode.IsLetter(r) && r < 0x2E80 {
			latin++
		}
	}
	return cjk > latin
}

// bridgeScopedKB reports whether any KB of the search is in the bridge scope.
// The bridge acts on the query, which is shared by every KB in the search, so
// a mixed scope (CJK corpus + Latin corpus) is bridged when any member is
// listed; callers wanting per-KB isolation should keep mixed scopes out of
// the list rather than rely on partial translation.
func bridgeScopedKB(kbs []*types.KnowledgeBase, cfg *types.CrossLanguageBridgeConfig) bool {
	if len(cfg.KBIDs) == 0 {
		return true
	}
	for _, kb := range kbs {
		if kb == nil {
			continue
		}
		for _, id := range cfg.KBIDs {
			if kb.ID == id {
				return true
			}
		}
	}
	return false
}

// applyCrossLanguageBridge rewrites params.QueryText to its English
// translation when the bridge is enabled for this search, the query is
// CJK-dominant, and a chat model can translate it. Every failure path keeps
// the original query: the bridge is an enhancement, never a gate - a corpus
// that cannot be translated still gets the retrieval it always had.
//
// A caller-supplied QueryEmbedding was computed from the original text, so a
// successful bridge clears it to force re-embedding of the translated query;
// otherwise the keyword leg would search English while the vector leg kept
// searching Chinese/Japanese.
func (s *knowledgeBaseService) applyCrossLanguageBridge(ctx context.Context,
	kbs []*types.KnowledgeBase, params *types.SearchParams, cfg *types.CrossLanguageBridgeConfig,
) {
	if params.QueryText == "" || !cjkDominantQuery(params.QueryText) {
		return
	}
	if !bridgeScopedKB(kbs, cfg) {
		return
	}

	modelID := cfg.ModelID
	if modelID == "" {
		models, err := s.modelService.ListModels(ctx)
		if err != nil {
			logger.Warnf(ctx, "Cross-language bridge skipped: cannot list models: %v", err)
			return
		}
		m := types.PickChatModel(models, "")
		if m == nil {
			logger.Warnf(ctx, "Cross-language bridge skipped: workspace has no active chat model")
			return
		}
		modelID = m.ID
	}
	chatModel, err := s.modelService.GetChatModel(ctx, modelID)
	if err != nil || chatModel == nil {
		logger.Warnf(ctx, "Cross-language bridge skipped: chat model %s unavailable: %v", modelID, err)
		return
	}

	thinking := false
	modelCtx := types.WithLLMCallMetadata(ctx, "cross_language_bridge", "")
	resp, err := chatModel.Chat(modelCtx, []chat.Message{
		{Role: "system", Content: bridgeTranslateSystemPrompt},
		{Role: "user", Content: params.QueryText},
	}, &chat.ChatOptions{
		Temperature:         0,
		MaxCompletionTokens: crossLanguageBridgeMaxTokens,
		Thinking:            &thinking,
	})
	if err != nil {
		logger.Warnf(ctx, "Cross-language bridge failed, searching the original query: %v", err)
		return
	}
	translated := strings.TrimSpace(resp.Content)
	if translated == "" {
		logger.Warnf(ctx, "Cross-language bridge returned empty translation, searching the original query")
		return
	}

	original := params.QueryText
	if cfg.Mode == "concat" {
		params.QueryText = original + " " + translated
	} else {
		params.QueryText = translated
	}
	params.QueryEmbedding = nil

	logger.Infof(ctx,
		"Cross-language bridge applied, mode=%s model=%s, original: %s, bridged: %s",
		cfg.Mode, modelID,
		secutils.SanitizeForLog(original), secutils.SanitizeForLog(params.QueryText))
}
