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
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.service.handler.InferenceHandler;
import io.gravitee.inference.service.handler.InferenceHandlerFactory;
import io.gravitee.inference.service.handler.VllmInferenceHandlerFactory;
import io.gravitee.inference.service.model.VllmModelFactory;
import io.gravitee.inference.service.repository.HandlerRepository;
import io.gravitee.inference.vllm.VllmConfig;
import io.reactivex.rxjava3.core.Single;
import io.vertx.rxjava3.core.Vertx;
import java.nio.file.Path;
import java.util.Map;

/**
 * Provider for the VLLM inference format.
 *
 * <p>Unlike LlamaCppProvider, this provider does NOT download models via Java.
 * vLLM handles model downloading internally through its Python engine, using
 * HuggingFace model identifiers (e.g. "Qwen/Qwen3-0.6B").
 */
public class VllmProvider implements InferenceHandlerProvider {

  private final VllmInferenceHandlerFactory handlerFactory;
  private final Path venvPath;

  public VllmProvider(Vertx vertx, String venvPath) {
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

    return Single.fromCallable(() -> {
      VllmConfig vllmConfig = buildVllmConfig(config);
      return repository.add(handlerFactory.create(vllmConfig));
    });
  }

  @Override
  public InferenceHandlerFactory<?> factory() {
    return handlerFactory;
  }

  private VllmConfig buildVllmConfig(ConfigWrapper config) {
    // The model is the HuggingFace model ID — no file path needed
    String model = config.get(Constants.MODEL_PATH);
    if (model == null || model.isBlank()) {
      // Fallback to modelName
      model = config.get(Constants.MODEL_NAME);
    }
    if (model == null || model.isBlank()) {
      throw new IllegalArgumentException(
        "model (modelPath or modelName) is required for VLLM format"
      );
    }

    Map<String, Object> modelParams = config.get(
      Constants.MODEL_PARAMS,
      Map.of()
    );
    Map<String, Object> context = config.get(Constants.CONTEXT, Map.of());

    String dtype = stringValue(modelParams.get("dtype"), "auto");
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

    return new VllmConfig(
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
      venvPath
    );
  }

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
}
