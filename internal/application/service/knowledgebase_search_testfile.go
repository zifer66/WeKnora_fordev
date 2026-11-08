package service

import (
	"context"
	"regexp"
	"slices"
	"strings"

	"github.com/Tencent/WeKnora/internal/logger"
	"github.com/Tencent/WeKnora/internal/types"
)

// testFilePattern matches the document title of a test file across common
// languages: path segments (tests/, __tests__/, spec/), Python (test_*.py,
// *_test.py), Go (*_test.go), Rust (*_tests.rs) and Java (*Test.java,
// *Tests.java, *IT.java). Down-weighting, not exclusion: a strong match can
// still surface, and the penalty is a multiplier so the fused ranking keeps
// its meaning.
var testFilePattern = regexp.MustCompile(
	`(^|/)(tests?|__tests__|spec)(/|__)` +
		`|(^|/)test_[^/]+\.py$` +
		`|(^|/)[^/]+_test\.py$` +
		`|(^|/)[^/]+_test\.go$` +
		`|(^|/)[^/]+_tests?\.rs$` +
		`|(Test|Tests|IT)\.java$`)

// testFileQueryPattern matches a query that is itself about tests. Those
// questions legitimately want test files at the top, so the penalty is
// skipped for them; without this gate the penalty cost 1-2 top-1 hits on
// golden sets whose gold answers include test files.
var testFileQueryPattern = regexp.MustCompile(
	`(?i)\btest|测试|試験|conftest|fixture|pytest|unittest|spec\b`)

// testFileTitle normalizes a document title for the test-file pattern.
// Symbol-level code ingestion names documents "path/to/file.go::symbol";
// the pattern's end anchors must see the file path, not the symbol.
func testFileTitle(title string) string {
	if i := strings.LastIndex(title, "::"); i >= 0 {
		return title[:i]
	}
	return title
}

// applyTestFilePenalty multiplies the fused score of chunks whose document is
// a test file by penalty and re-sorts, in place semantics of the fusion step
// before it. It is a no-op when the query asks about tests, when the penalty
// is a no-op multiplier, or when no chunk's document is a test file, so the
// common configuration (feature off, or non-code corpora) pays nothing.
//
// titles maps knowledge ID to document title; chunks whose document title
// cannot be resolved are left unchanged. kbIDs scopes the penalty by chunk
// knowledge base; empty applies to every KB.
func applyTestFilePenalty(chunks []*types.IndexWithScore, titles map[string]string,
	query string, penalty float64, kbIDs []string,
) []*types.IndexWithScore {
	if len(chunks) == 0 || penalty <= 0 || penalty >= 1 {
		return chunks
	}
	if query != "" && testFileQueryPattern.MatchString(query) {
		return chunks
	}

	downweighted := 0
	for _, chunk := range chunks {
		if chunk == nil || len(kbIDs) > 0 && !slices.Contains(kbIDs, chunk.KnowledgeBaseID) {
			continue
		}
		title, ok := titles[chunk.KnowledgeID]
		if !ok || !testFilePattern.MatchString(testFileTitle(title)) {
			continue
		}
		chunk.Score *= penalty
		downweighted++
	}
	if downweighted == 0 {
		return chunks
	}
	slices.SortFunc(chunks, sortByScoreDesc)
	return chunks
}

// downweightTestFiles fetches the document titles of the fused chunks and
// applies the test-file penalty. A title-fetch failure keeps the fused order:
// like the rerank fallback, a degraded enhancement must not degrade retrieval
// below its no-feature baseline.
func (s *knowledgeBaseService) downweightTestFiles(ctx context.Context,
	query string, chunks []*types.IndexWithScore, cfg *types.TestFilePenaltyConfig,
) []*types.IndexWithScore {
	if len(chunks) == 0 {
		return chunks
	}

	knowledgeIDs := make([]string, 0, len(chunks))
	seen := make(map[string]bool, len(chunks))
	for _, chunk := range chunks {
		if chunk == nil || chunk.KnowledgeID == "" || seen[chunk.KnowledgeID] {
			continue
		}
		if len(cfg.KBIDs) > 0 && !slices.Contains(cfg.KBIDs, chunk.KnowledgeBaseID) {
			continue
		}
		seen[chunk.KnowledgeID] = true
		knowledgeIDs = append(knowledgeIDs, chunk.KnowledgeID)
	}
	if len(knowledgeIDs) == 0 {
		return chunks
	}

	tenantID := types.MustTenantIDFromContext(ctx)
	knowledgeMap, err := s.fetchKnowledgeDataWithShared(ctx, tenantID, knowledgeIDs)
	if err != nil {
		logger.Warnf(ctx, "Test-file penalty skipped: cannot fetch knowledge titles: %v", err)
		return chunks
	}
	titles := make(map[string]string, len(knowledgeMap))
	for id, k := range knowledgeMap {
		if k != nil {
			titles[id] = k.Title
		}
	}
	return applyTestFilePenalty(chunks, titles, query, cfg.Penalty, cfg.KBIDs)
}
