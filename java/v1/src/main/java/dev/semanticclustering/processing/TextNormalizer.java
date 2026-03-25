package dev.semanticclustering.processing;

import dev.semanticclustering.config.TextNormalizeConfig;

import java.text.Normalizer;

/**
 * Configurable text cleanup before embedding.
 */
public final class TextNormalizer {
    private final TextNormalizeConfig config;

    public TextNormalizer(TextNormalizeConfig config) {
        this.config = config;
    }

    public String normalize(String text) {
        if (text == null) {
            return "";
        }
        if (!config.enabled) {
            return text;
        }
        String normalized = text.trim();
        if (config.unicodeNfc) {
            normalized = Normalizer.normalize(normalized, Normalizer.Form.NFC);
        }
        if (config.removeControlCharacters) {
            normalized = normalized.replaceAll("\\p{Cc}+", " ");
        }
        if (config.collapseWhitespace) {
            normalized = normalized.replaceAll("\\s+", " ");
        }
        if (config.lowercase) {
            normalized = normalized.toLowerCase();
        }
        return normalized;
    }
}
