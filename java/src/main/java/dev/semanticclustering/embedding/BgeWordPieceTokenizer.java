package dev.semanticclustering.embedding;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * WordPiece vocabulary tokenization for BERT-style ONNX sentence models.
 */
public final class BgeWordPieceTokenizer {
    private static final String CLS = "[CLS]";
    private static final String SEP = "[SEP]";
    private static final String PAD = "[PAD]";
    private static final String UNK = "[UNK]";

    private final Map<String, Integer> vocab;
    private final int maxSequenceLength;

    public BgeWordPieceTokenizer(Path vocabPath, int maxSequenceLength) throws IOException {
        this.vocab = loadVocabulary(vocabPath);
        this.maxSequenceLength = maxSequenceLength;
        for (String token : List.of(CLS, SEP, PAD, UNK)) {
            if (!vocab.containsKey(token)) {
                throw new IllegalArgumentException("Tokenizer vocabulary missing token: " + token);
            }
        }
    }

    public TokenizedInput encode(String text) {
        List<String> basicTokens = basicTokenize(text == null ? "" : text);
        List<String> pieces = new ArrayList<>();
        for (String token : basicTokens) {
            pieces.addAll(wordPieceTokenize(token));
        }

        int maxContent = maxSequenceLength - 2;
        if (pieces.size() > maxContent) {
            pieces = pieces.subList(0, maxContent);
        }

        List<String> tokens = new ArrayList<>(maxSequenceLength);
        tokens.add(CLS);
        tokens.addAll(pieces);
        tokens.add(SEP);

        long[] inputIds = new long[maxSequenceLength];
        long[] attentionMask = new long[maxSequenceLength];
        long[] tokenTypeIds = new long[maxSequenceLength];
        int cursor = 0;
        for (; cursor < tokens.size(); cursor++) {
            inputIds[cursor] = vocabId(tokens.get(cursor));
            attentionMask[cursor] = 1;
        }
        int padId = vocabId(PAD);
        for (; cursor < maxSequenceLength; cursor++) {
            inputIds[cursor] = padId;
        }
        return new TokenizedInput(inputIds, attentionMask, tokenTypeIds);
    }

    private List<String> basicTokenize(String text) {
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
                .toLowerCase(Locale.ROOT)
                .replaceAll("\\s+", " ")
                .trim();
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < normalized.length(); i++) {
            char character = normalized.charAt(i);
            if (Character.isLetterOrDigit(character)) {
                current.append(character);
                continue;
            }
            if (!current.isEmpty()) {
                tokens.add(current.toString());
                current.setLength(0);
            }
            if (!Character.isWhitespace(character)) {
                tokens.add(String.valueOf(character));
            }
        }
        if (!current.isEmpty()) {
            tokens.add(current.toString());
        }
        return tokens;
    }

    private List<String> wordPieceTokenize(String token) {
        if (vocab.containsKey(token)) {
            return List.of(token);
        }
        List<String> pieces = new ArrayList<>();
        int start = 0;
        while (start < token.length()) {
            int end = token.length();
            String current = null;
            while (start < end) {
                String candidate = token.substring(start, end);
                if (start > 0) {
                    candidate = "##" + candidate;
                }
                if (vocab.containsKey(candidate)) {
                    current = candidate;
                    break;
                }
                end -= 1;
            }
            if (current == null) {
                return List.of(UNK);
            }
            pieces.add(current);
            start = end;
        }
        return pieces;
    }

    private int vocabId(String token) {
        return vocab.getOrDefault(token, vocab.get(UNK));
    }

    private static Map<String, Integer> loadVocabulary(Path path) throws IOException {
        Map<String, Integer> map = new HashMap<>();
        List<String> lines = Files.readAllLines(path);
        for (int i = 0; i < lines.size(); i++) {
            map.put(lines.get(i).trim(), i);
        }
        return Map.copyOf(map);
    }
}
