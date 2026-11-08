package service

import (
	"testing"

	"github.com/Tencent/WeKnora/internal/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTestFilePattern(t *testing.T) {
	t.Parallel()

	cases := []struct {
		title string
		want  bool
	}{
		// Java conventions measured in the research corpus.
		{"gson/src/test/java/com/google/gson/FooTest.java", true},
		{"gson/src/test/java/com/google/gson/FooTests.java", true},
		{"gson/src/test/java/com/google/gson/FooIT.java", true},
		{"gson/src/main/java/com/google/gson/Foo.java", false},
		// Symbol-level ingestion titles keep the path before "::".
		{"gson/src/test/java/com/google/gson/FooTest.java::testBar", true},
		{"gson/src/main/java/com/google/gson/Foo.java::bar", false},
		{"src/main/java/Foo.java::TestSomething", false},
		// Path segments and other languages.
		{"tests/conftest.py", true},
		{"src/__tests__/app.spec.ts", true},
		{"spec/helper.rb", true},
		{"pkg/parse_test.go", true},
		{"scripts/test_lint.sh", false},
		{"app/models/user_test.py", true},
		{"src/lib.rs", false},
		{"src/lib_tests.rs", true},
		// A file merely containing the word test must not match.
		{"latest_release_notes.md", false},
		{"attestation.md", false},
		{"contest_rules.txt", false},
	}
	for _, tc := range cases {
		t.Run(tc.title, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.want, testFilePattern.MatchString(testFileTitle(tc.title)))
		})
	}
}

func penaltyChunks() []*types.IndexWithScore {
	// Fused order: the lexically similar test chunk outranks the
	// implementation, the exact shape measured on the golden set.
	return []*types.IndexWithScore{
		{ChunkID: "c1", KnowledgeID: "k-test", KnowledgeBaseID: "kb-code", Score: 0.9},
		{ChunkID: "c2", KnowledgeID: "k-impl", KnowledgeBaseID: "kb-code", Score: 0.8},
		{ChunkID: "c3", KnowledgeID: "k-doc", KnowledgeBaseID: "kb-other", Score: 0.4},
	}
}

func penaltyTitles() map[string]string {
	return map[string]string{
		"k-test": "gson/src/test/java/com/google/gson/FooTest.java::testBar",
		"k-impl": "gson/src/main/java/com/google/gson/Foo.java::bar",
		"k-doc":  "gson/src/test/java/other/AlsoTest.java::x",
	}
}

func TestApplyTestFilePenalty(t *testing.T) {
	t.Parallel()

	t.Run("demotes test chunks below the implementation", func(t *testing.T) {
		t.Parallel()
		chunks := penaltyChunks()
		out := applyTestFilePenalty(chunks, penaltyTitles(),
			"where is the date format handled", 0.5, nil)

		require.Len(t, out, 3)
		assert.Equal(t, "c2", out[0].ChunkID, "implementation should outrank the demoted test chunk")
		assert.Equal(t, "c1", out[1].ChunkID)
		assert.InDelta(t, 0.45, out[1].Score, 1e-9)
		assert.InDelta(t, 0.8, out[0].Score, 1e-9)
	})

	t.Run("kb scoping limits the penalty", func(t *testing.T) {
		t.Parallel()
		chunks := penaltyChunks()
		out := applyTestFilePenalty(chunks, penaltyTitles(),
			"where is the date format handled", 0.5, []string{"kb-other"})

		require.Len(t, out, 3)
		assert.Equal(t, "c1", out[0].ChunkID, "kb-code chunks stay untouched")
		assert.InDelta(t, 0.9, out[0].Score, 1e-9)
		assert.InDelta(t, 0.2, out[2].Score, 1e-9, "kb-other test chunk is demoted")
	})

	t.Run("query about tests skips the penalty", func(t *testing.T) {
		t.Parallel()
		for _, q := range []string{
			"which test covers date formats",
			"日期格式化的测试在哪里",
			"conftest で何をしている",
			"how does the spec fixture work",
		} {
			chunks := penaltyChunks()
			out := applyTestFilePenalty(chunks, penaltyTitles(), q, 0.5, nil)
			assert.Equal(t, "c1", out[0].ChunkID, "query %q", q)
			assert.InDelta(t, 0.9, out[0].Score, 1e-9, "query %q", q)
		}
	})

	t.Run("no-op multipliers and missing titles change nothing", func(t *testing.T) {
		t.Parallel()
		chunks := penaltyChunks()
		out := applyTestFilePenalty(chunks, penaltyTitles(), "impl question", 1.0, nil)
		assert.InDelta(t, 0.9, out[0].Score, 1e-9)

		chunks = penaltyChunks()
		out = applyTestFilePenalty(chunks, map[string]string{}, "impl question", 0.5, nil)
		assert.InDelta(t, 0.9, out[0].Score, 1e-9)
	})
}
