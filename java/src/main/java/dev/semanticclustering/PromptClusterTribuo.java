package dev.semanticclustering;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ClusteringFactory;
import org.tribuo.clustering.hdbscan.HdbscanModel;
import org.tribuo.clustering.hdbscan.HdbscanTrainer;
import org.tribuo.impl.ArrayExample;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.awt.Color;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * Baseline: hash-based text features → Tribuo HDBSCAN* → 2D random projection for plotting (XChart).
 * Outliers use Tribuo cluster id 0 (noise). Valid clusters use positive ids.
 */
public final class PromptClusterTribuo {

    private static final int EMBED_DIM = 64;
    private static final long PROJ_SEED = 42L;
    private static final Pattern LIST_PREFIX = Pattern.compile("^\\d+\\.\\s*");
    private static final String TAG = "[JAVA|Tribuo]";
    private static final int BANNER_WIDTH = 76;

    static {
        Logger.getLogger("org.tribuo.clustering.hdbscan.HdbscanTrainer").setLevel(Level.WARNING);
    }

    private PromptClusterTribuo() {
    }

    private static void log(String msg) {
        System.out.println(TAG + " " + msg);
    }

    private static void javaBanner() {
        String line = "-".repeat(BANNER_WIDTH);
        System.out.println();
        System.out.println(line);
        System.out.println("  JAVA — Oracle Tribuo (HDBSCAN*) · hashed word features · 2D projection plot");
        System.out.println("  Noise points: cluster id 0");
        System.out.println(line);
    }

    public static void main(String[] args) throws IOException {
        Path promptsPath;
        Path out;
        if (args.length >= 2) {
            promptsPath = Path.of(args[0]);
            out = Path.of(args[1]);
        } else if (args.length == 1) {
            promptsPath = defaultPromptsPath();
            out = Path.of(args[0]);
        } else {
            promptsPath = defaultPromptsPath();
            out = Path.of(System.getProperty("user.dir")).resolve("../outputs/java_tribuo_hdbscan.png").normalize();
        }
        javaBanner();
        log("Prompts file: " + promptsPath.toAbsolutePath());
        List<String> data = loadPrompts(promptsPath);
        log("Loaded " + data.size() + " processed prompt(s)");
        Files.createDirectories(out.getParent());

        ClusteringFactory factory = new ClusteringFactory();
        MutableDataset<ClusterID> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("prompts", OffsetDateTime.now(), factory),
            factory
        );
        String[] names = featureNames(EMBED_DIM);
        List<double[]> vectors = new ArrayList<>();
        for (String text : data) {
            double[] v = textFeatures(text, EMBED_DIM);
            vectors.add(v);
            dataset.add(new ArrayExample<>(factory.getUnknownOutput(), names, v));
        }

        double[][] proj = projectionMatrix(EMBED_DIM, PROJ_SEED);
        List<Double> px = new ArrayList<>();
        List<Double> py = new ArrayList<>();
        for (double[] v : vectors) {
            double[] xy = project2d(v, proj);
            px.add(xy[0]);
            py.add(xy[1]);
        }

        // Align with Python script: small clusters, kd-tree style NN queries.
        HdbscanTrainer trainer = new HdbscanTrainer(
            2,
            DistanceType.L2.getDistance(),
            2,
            1,
            NeighboursQueryFactoryType.KD_TREE
        );
        log("Training HDBSCAN* (min cluster size 2, k=2, KD-tree NN)…");
        HdbscanModel model = trainer.train(dataset);
        List<Integer> labels = model.getClusterLabels();

        Map<Integer, List<Double>> mapX = new HashMap<>();
        Map<Integer, List<Double>> mapY = new HashMap<>();
        for (int i = 0; i < labels.size(); i++) {
            int label = labels.get(i);
            mapX.computeIfAbsent(label, k -> new ArrayList<>()).add(px.get(i));
            mapY.computeIfAbsent(label, k -> new ArrayList<>()).add(py.get(i));
        }

        XYChart chart = new XYChartBuilder()
            .width(960)
            .height(640)
            .title("Prompt Intent Clustering")
            .xAxisTitle("X")
            .yAxisTitle("Y")
            .build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setLegendVisible(true);
        chart.getStyler().setMarkerSize(10);

        Queue<Color> colors = new ArrayDeque<>(List.of(
            Color.CYAN, Color.GREEN, Color.MAGENTA, Color.ORANGE, Color.BLUE,
            new Color(0.4f, 0.7f, 0.2f), new Color(0.6f, 0.2f, 0.8f)
        ));

        List<Integer> keys = new ArrayList<>(mapX.keySet());
        keys.sort(Integer::compareTo);
        for (Integer clusterId : keys) {
            Color color = clusterId == 0 ? Color.RED : colors.poll();
            if (color == null) {
                color = Color.DARK_GRAY;
            }
            String seriesName = clusterId == 0 ? "noise (0)" : "cluster " + clusterId;
            var xs = mapX.get(clusterId).stream().mapToDouble(Double::doubleValue).toArray();
            var ys = mapY.get(clusterId).stream().mapToDouble(Double::doubleValue).toArray();
            var series = chart.addSeries(seriesName, xs, ys);
            series.setMarkerColor(color);
            series.setMarker(SeriesMarkers.CIRCLE);
        }

        BitmapEncoder.saveBitmap(chart, out.toAbsolutePath().toString(), BitmapEncoder.BitmapFormat.PNG);
        Path jsonOut = siblingJson(out);
        writeClusteringJson(promptsPath, out, jsonOut, data, labels);
        log("Wrote image  " + out.toAbsolutePath());
        log("Wrote JSON   " + jsonOut.toAbsolutePath());
        log("Done.");
        System.out.println();
    }

    private static Path siblingJson(Path imagePath) {
        String fn = imagePath.getFileName().toString();
        String base = fn.endsWith(".png") ? fn.substring(0, fn.length() - 4) : fn;
        return imagePath.resolveSibling(base + ".json");
    }

    private static String clusterJsonKeyTribuo(int label) {
        if (label == 0) {
            return "noise";
        }
        return "cluster_" + label;
    }

    private static void writeClusteringJson(
        Path promptsPath,
        Path imageOut,
        Path jsonOut,
        List<String> data,
        List<Integer> labels
    ) throws IOException {
        Map<Integer, List<String>> byLabel = new HashMap<>();
        for (int i = 0; i < data.size(); i++) {
            byLabel.computeIfAbsent(labels.get(i), k -> new ArrayList<>()).add(data.get(i));
        }
        List<Integer> order = new ArrayList<>(byLabel.keySet());
        order.sort(Integer::compareTo);
        Map<String, List<String>> clusters = new LinkedHashMap<>();
        for (int lab : order) {
            clusters.put(clusterJsonKeyTribuo(lab), byLabel.get(lab));
        }
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("program", "java-tribuo-hdbscan");
        root.put("prompts_file", promptsPath.toAbsolutePath().toString());
        root.put("output_image", imageOut.toAbsolutePath().toString());
        root.put("output_json", jsonOut.toAbsolutePath().toString());
        root.put("clusters", clusters);
        Gson gson = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();
        Files.writeString(jsonOut, gson.toJson(root), StandardCharsets.UTF_8);
    }

    private static Path defaultPromptsPath() {
        return Path.of(System.getProperty("user.dir")).resolve("../data/prompts.txt").normalize();
    }

    private static List<String> loadPrompts(Path path) throws IOException {
        if (!Files.isRegularFile(path)) {
            throw new IOException("Prompts file not found: " + path.toAbsolutePath());
        }
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        List<String> prompts = new ArrayList<>();
        for (String line : lines) {
            String s = line.strip();
            if (s.isEmpty() || s.startsWith("#")) {
                continue;
            }
            String p = processPromptLine(s);
            if (p != null) {
                prompts.add(p);
            }
        }
        if (prompts.isEmpty()) {
            throw new IOException("No prompts in file: " + path.toAbsolutePath());
        }
        return List.copyOf(prompts);
    }

    /** Strip list numbering, inline # comments, and digit runs (matches Python prompt_cluster). */
    private static String processPromptLine(String line) {
        String t = LIST_PREFIX.matcher(line.strip()).replaceFirst("").strip();
        int hash = t.indexOf('#');
        if (hash >= 0) {
            t = t.substring(0, hash).strip();
        }
        if (t.isEmpty()) {
            return null;
        }
        t = t.replaceAll("\\d+", " ").replaceAll("\\s+", " ").strip();
        return t.isEmpty() ? null : t;
    }

    private static String[] featureNames(int dim) {
        String[] n = new String[dim];
        for (int i = 0; i < dim; i++) {
            n[i] = "f" + i;
        }
        return n;
    }

    private static double[] textFeatures(String text, int dim) {
        double[] v = new double[dim];
        for (String w : text.toLowerCase().split("\\W+")) {
            if (w.isEmpty()) {
                continue;
            }
            int h = Math.floorMod(w.hashCode(), dim);
            v[h] += 1.0;
        }
        double norm = 0.0;
        for (double x : v) {
            norm += x * x;
        }
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < dim; i++) {
                v[i] /= norm;
            }
        }
        return v;
    }

    private static double[][] projectionMatrix(int dim, long seed) {
        Random rnd = new Random(seed);
        double[][] m = new double[dim][2];
        for (int i = 0; i < dim; i++) {
            m[i][0] = rnd.nextGaussian();
            m[i][1] = rnd.nextGaussian();
        }
        return m;
    }

    private static double[] project2d(double[] v, double[][] proj) {
        double x = 0.0;
        double y = 0.0;
        for (int i = 0; i < v.length; i++) {
            x += v[i] * proj[i][0];
            y += v[i] * proj[i][1];
        }
        return new double[]{x, y};
    }
}
