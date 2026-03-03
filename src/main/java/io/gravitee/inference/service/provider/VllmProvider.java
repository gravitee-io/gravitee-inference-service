/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.inference.service.provider;

import static io.gravitee.inference.api.Constants.INFERENCE_TYPE;

import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.memory.MemoryCheckPolicy;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.service.handler.InferenceHandler;
import io.gravitee.inference.service.handler.InferenceHandlerFactory;
import io.gravitee.inference.service.handler.VllmInferenceHandlerFactory;
import io.gravitee.inference.service.model.VllmModelFactory;
import io.gravitee.inference.service.repository.HandlerRepository;
import io.gravitee.inference.vllm.VllmConfig;
import io.gravitee.reactive.webclient.api.SafetensorsInfo;
import io.gravitee.reactive.webclient.huggingface.downloader.HuggingFaceDownloader;
import io.reactivex.rxjava3.core.Single;
import io.vertx.rxjava3.core.RxHelper;
import io.vertx.rxjava3.core.Vertx;
import java.nio.file.Path;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Provider for the VLLM inference format.
 *
 * <p>Unlike LlamaCppProvider, this provider does NOT download models via Java.
 * vLLM handles model downloading internally through its Python engine, using
 * HuggingFace model identifiers (e.g. "Qwen/Qwen3-0.6B").
 */
public class VllmProvider implements InferenceHandlerProvider {

  private static final Logger LOGGER = LoggerFactory.getLogger(
    VllmProvider.class
  );

  private final VllmInferenceHandlerFactory handlerFactory;
  private final Path venvPath;
  private final Vertx vertx;

  public VllmProvider(Vertx vertx, String venvPath) {
    this.vertx = vertx;
    this.handlerFactory = new VllmInferenceHandlerFactory(
      vertx,
      new VllmModelFactory()
    );
    this.venvPath = venvPath != null && !venvPath.isBlank()
      ? Path.of(venvPath)
      : null;
  }

  @Override
  public Single<InferenceHandler> provide(
    InferenceRequest inferenceRequest,
    HandlerRepository repository
  ) {
    var config = new ConfigWrapper(inferenceRequest.payload());
    InferenceType type = InferenceType.valueOf(config.get(INFERENCE_TYPE));
    if (type != InferenceType.TEXT_GENERATION) {
      return Single.error(
        new IllegalArgumentException(
          "Unsupported inference type '" + type + "' for format VLLM"
        )
      );
    }

    return buildVllmConfig(config)
      .observeOn(RxHelper.blockingScheduler(vertx))
      .map(vllmConfig -> repository.add(handlerFactory.create(vllmConfig)));
  }

  @Override
  public InferenceHandlerFactory<?> factory() {
    return handlerFactory;
  }

  private Single<VllmConfig> buildVllmConfig(ConfigWrapper config) {
    // The model is the HuggingFace model ID — no file path needed
    String modelId = config.get(Constants.MODEL_PATH);
    if (modelId == null || modelId.isBlank()) {
      // Fallback to modelName
      modelId = config.get(Constants.MODEL_NAME);
    }
    if (modelId == null || modelId.isBlank()) {
      return Single.error(
        new IllegalArgumentException(
          "model (modelPath or modelName) is required for VLLM format"
        )
      );
    }

    final String model = modelId;

    Map<String, Object> modelParams = config.get(
      Constants.MODEL_PARAMS,
      Map.of()
    );
    Map<String, Object> context = config.get(Constants.CONTEXT, Map.of());

    String dtype = stringValue(modelParams.get("dtype"), "auto");
    String hfToken = stringValue(modelParams.get("hfToken"), null);
    int maxModelLen = intValue(context.get("maxModelLen"), 0);
    int maxNumSeqs = intValue(context.get(Constants.CONTEXT_N_SEQ_MAX), 8);
    double gpuMemoryUtilization = doubleValue(
      modelParams.get("gpuMemoryUtilization"),
      0
    );
    int maxNumBatchedTokens = intValue(
      modelParams.get("maxNumBatchedTokens"),
      0
    );
    boolean enforceEager = booleanValue(modelParams.get("enforceEager"), false);
    boolean trustRemoteCode = booleanValue(
      modelParams.get("trustRemoteCode"),
      false
    );
    String quantization = stringValue(modelParams.get("quantization"), null);
    double swapSpace = doubleValue(modelParams.get("swapSpace"), 0);
    Integer seed = modelParams.get("seed") != null
      ? intValue(modelParams.get("seed"), null)
      : null;
    boolean enablePrefixCaching = booleanValue(
      modelParams.get("enablePrefixCaching"),
      false
    );
    boolean enableChunkedPrefill = booleanValue(
      modelParams.get("enableChunkedPrefill"),
      false
    );
    String kvCacheDtype = stringValue(modelParams.get("kvCacheDtype"), null);
    boolean enableLora = booleanValue(modelParams.get("enableLora"), false);
    int maxLoras = intValue(modelParams.get("maxLoras"), 0);
    int maxLoraRank = intValue(modelParams.get("maxLoraRank"), 0);

    // Fetch model metadata from HuggingFace Hub reactively, then build config
    return fetchModelMetadata(model, dtype, hfToken).map(meta ->
      new VllmConfig(
        model,
        dtype,
        maxModelLen,
        maxNumSeqs,
        gpuMemoryUtilization,
        maxNumBatchedTokens,
        enforceEager,
        trustRemoteCode,
        quantization,
        swapSpace,
        seed,
        enablePrefixCaching,
        enableChunkedPrefill,
        kvCacheDtype,
        enableLora,
        maxLoras,
        maxLoraRank,
        venvPath,
        enumValue(
          MemoryCheckPolicy.class,
          modelParams.get("memoryCheckPolicy"),
          MemoryCheckPolicy.WARN
        ),
        meta.totalParams(),
        meta.bytesPerParam(),
        meta.numHiddenLayers(),
        meta.numKvHeads(),
        meta.headDim(),
        meta.multimodal(),
        hfToken
      )
    );
  }

  // ── HuggingFace model metadata resolution ─────────────────────────────

  /**
   * Fetches model metadata from the HuggingFace Hub API for VRAM estimation.
   *
   * <p>Makes two reactive HTTP calls in parallel (soft-fail on both):
   * <ol>
   *   <li>{@code GET /api/models/{model}} → safetensors total + dtype breakdown</li>
   *   <li>{@code GET /{model}/resolve/main/config.json} → architecture dims</li>
   * </ol>
   *
   * <p>On any failure, returns zeros — the memory estimator will return
   * {@link io.gravitee.inference.api.memory.MemoryEstimate#unknown()} and
   * the check is silently skipped.
   */
  private Single<ModelMetadata> fetchModelMetadata(
    String model,
    String dtype,
    String hfToken
  ) {
    HuggingFaceDownloader hf = hfToken != null
      ? HuggingFaceDownloader.withToken(vertx, hfToken)
      : new HuggingFaceDownloader(vertx);

    // 1. Fetch safetensors info → totalParams + bytesPerParam
    Single<long[]> weightInfo = hf
      .fetchModelInfo(model)
      .map(info -> {
        if (info.hasSafetensorsInfo()) {
          SafetensorsInfo st = info.safetensors();
          long total = st.total();
          long weightBytes = st.estimateWeightBytes(dtype);
          int bpp = total > 0 ? (int) (weightBytes / total) : 2;
          return new long[] { total, bpp };
        }
        return new long[] { 0, 2 };
      })
      .onErrorReturn(e -> {
        LOGGER.warn(
          "Could not fetch model info for [{}]: {} — VRAM estimation will be skipped",
          model,
          e.getMessage()
        );
        return new long[] { 0, 2 };
      });

    // 2. Fetch config.json → architecture dimensions
    Single<int[]> archInfo = hf
      .fetchFileAsJson(model, "config.json")
      .map(configJson -> {
        int layers = configJson.getInteger("num_hidden_layers", 0);
        int kvHeads = configJson.getInteger(
          "num_key_value_heads",
          configJson.getInteger("num_attention_heads", 0)
        );
        int hiddenSize = configJson.getInteger("hidden_size", 0);
        int numAttnHeads = configJson.getInteger("num_attention_heads", 0);
        int hd = configJson.getInteger(
          "head_dim",
          numAttnHeads > 0 ? hiddenSize / numAttnHeads : 0
        );
        boolean mm =
          configJson.containsKey("vision_config") ||
          configJson.containsKey("audio_config");
        return new int[] { layers, kvHeads, hd, mm ? 1 : 0 };
      })
      .onErrorReturn(e -> {
        LOGGER.warn(
          "Could not fetch config.json for [{}]: {} — KV-cache estimation will be skipped",
          model,
          e.getMessage()
        );
        return new int[] { 0, 0, 0, 0 };
      });

    // Zip both in parallel → ModelMetadata
    return Single.zip(weightInfo, archInfo, (w, a) ->
      new ModelMetadata(w[0], (int) w[1], a[0], a[1], a[2], a[3] == 1)
    );
  }

  private record ModelMetadata(
    long totalParams,
    int bytesPerParam,
    int numHiddenLayers,
    int numKvHeads,
    int headDim,
    boolean multimodal
  ) {}

  private String stringValue(Object value, String defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    String s = value.toString().trim();
    return s.isEmpty() ? defaultValue : s;
  }

  private boolean booleanValue(Object value, boolean defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Boolean b) {
      return b;
    }
    return Boolean.parseBoolean(value.toString());
  }

  private int intValue(Object value, Integer defaultValue) {
    if (value == null) {
      return defaultValue != null ? defaultValue : 0;
    }
    if (value instanceof Number number) {
      return number.intValue();
    }
    return Integer.parseInt(value.toString());
  }

  private double doubleValue(Object value, double defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number number) {
      return number.doubleValue();
    }
    return Double.parseDouble(value.toString());
  }

  private <T extends Enum<T>> T enumValue(
    Class<T> type,
    Object value,
    T defaultValue
  ) {
    if (value == null) return defaultValue;
    String name = value.toString().trim();
    if (name.isEmpty()) return defaultValue;
    try {
      return Enum.valueOf(type, name.toUpperCase());
    } catch (IllegalArgumentException e) {
      return defaultValue;
    }
  }
}
