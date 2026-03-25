package dev.semanticclustering.config;

/** Text normalization flags for the embedding / clustering pipeline. */
public final class TextNormalizeConfig {
    public boolean enabled = true;
    public boolean lowercase = false;
    public boolean collapseWhitespace = true;
    public boolean unicodeNfc = true;
    public boolean removeControlCharacters = true;

    public static TextNormalizeConfig defaults() {
        return new TextNormalizeConfig();
    }
}
