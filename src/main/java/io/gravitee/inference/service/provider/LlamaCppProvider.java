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
import io.gravitee.inference.llama.cpp.ModelConfig;
import io.gravitee.inference.service.handler.InferenceHandler;
import io.gravitee.inference.service.handler.InferenceHandlerFactory;
import io.gravitee.inference.service.handler.LlamaCppInferenceHandlerFactory;
import io.gravitee.inference.service.model.LlamaCppModelFactory;
import io.gravitee.inference.service.repository.HandlerRepository;
import io.gravitee.llama.cpp.AttentionType;
import io.gravitee.llama.cpp.FlashAttentionType;
import io.gravitee.llama.cpp.LlamaLogLevel;
import io.gravitee.llama.cpp.PoolingType;
import io.gravitee.llama.cpp.SplitMode;
import io.gravitee.reactive.webclient.api.FetchModelConfig;
import io.gravitee.reactive.webclient.api.ModelFetcher;
import io.gravitee.reactive.webclient.api.ModelFile;
import io.gravitee.reactive.webclient.api.ModelFileType;
import io.gravitee.reactive.webclient.huggingface.downloader.HuggingFaceDownloader;
import io.reactivex.rxjava3.core.Single;
import io.vertx.rxjava3.core.RxHelper;
import io.vertx.rxjava3.core.Vertx;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.UUID;

public class LlamaCppProvider implements InferenceHandlerProvider {

  private final Vertx vertx;
  private final Path baseModelPath;
  private final ModelFetcher modelFetcher;
  private final LlamaCppInferenceHandlerFactory handlerFactory;

  public LlamaCppProvider(Vertx vertx, String modelPath) {
    this.vertx = vertx;
    this.baseModelPath = Path.of(modelPath);
    this.modelFetcher = new HuggingFaceDownloader(vertx);
    this.handlerFactory = new LlamaCppInferenceHandlerFactory(vertx, new LlamaCppModelFactory());
  }

  @Override
  public Single<InferenceHandler> provide(InferenceRequest inferenceRequest, HandlerRepository repository) {
    var config = new ConfigWrapper(inferenceRequest.payload());
    InferenceType type = InferenceType.valueOf(config.get(INFERENCE_TYPE));
    if (type != InferenceType.TEXT_GENERATION) {
      return Single.error(new IllegalArgumentException("Unsupported inference type '" + type + "' for format LLAMA_CPP"));
    }

    return resolveModelPath(config)
      .flatMap(modelPath ->
        resolveLoraPath(config).map(loraPath -> buildModelConfig(config, modelPath, loraPath.orElse(null)))
      )
      .map(modelConfig -> repository.add(handlerFactory.create(modelConfig)));
  }

  @Override
  public InferenceHandlerFactory<?> factory() {
    return handlerFactory;
  }

  private Single<Path> resolveModelPath(ConfigWrapper config) {
    String modelPath = config.get(Constants.MODEL_PATH);
    if (modelPath == null || modelPath.isBlank()) {
      return Single.error(new IllegalArgumentException("modelPath is required"));
    }

    Path resolved = resolveModelPath(modelPath);
    if (Files.exists(resolved)) {
      return Single.just(resolved);
    }

    String modelName = config.get(Constants.MODEL_NAME, UUID.randomUUID().toString());
    FetchModelConfig fetchConfig = new FetchModelConfig(
      modelName,
      java.util.List.of(new ModelFile(modelPath, ModelFileType.MODEL)),
      getFileDirectory(modelName)
    );
    return modelFetcher
      .fetchModel(fetchConfig)
      .subscribeOn(RxHelper.blockingScheduler(vertx))
      .observeOn(RxHelper.blockingScheduler(vertx))
      .map(files -> Path.of(files.get(ModelFileType.MODEL)));
  }

  private ModelConfig buildModelConfig(ConfigWrapper config, Path modelPath, Path loraPath) {
    int availableProcessors = Math.max(1, Runtime.getRuntime().availableProcessors());
    Map<String, Object> context = config.get(Constants.CONTEXT, Map.of());
    Map<String, Object> modelParams = config.get(Constants.MODEL_PARAMS, Map.of());

    SplitMode splitMode = enumValue(SplitMode.class, modelParams.get(Constants.MODEL_SPLIT_MODE), SplitMode.NONE);
    PoolingType poolingType = enumValue(
      PoolingType.class,
      context.get(Constants.CONTEXT_POOLING_TYPE),
      PoolingType.UNSPECIFIED
    );
    AttentionType attentionType = enumValue(
      AttentionType.class,
      context.get(Constants.CONTEXT_ATTENTION_TYPE),
      AttentionType.UNSPECIFIED
    );
    FlashAttentionType flashAttnType = enumValue(
      FlashAttentionType.class,
      context.get(Constants.CONTEXT_FLASH_ATTN_TYPE),
      FlashAttentionType.UNSPECIFIED
    );
    LlamaLogLevel logLevel = enumValue(LlamaLogLevel.class, modelParams.get(Constants.MODEL_LOG_LEVEL), null);

    return new ModelConfig(
      modelPath,
      intValue(context.get(Constants.CONTEXT_N_CTX), 4096),
      intValue(context.get(Constants.CONTEXT_N_BATCH), 512),
      intValue(context.get(Constants.CONTEXT_N_UBATCH), 512),
      intValue(context.get(Constants.CONTEXT_N_SEQ_MAX), 8),
      availableProcessors,
      availableProcessors,
      intValue(modelParams.get(Constants.MODEL_N_GPU_LAYERS), 99),
      booleanValue(modelParams.get(Constants.MODEL_USE_MLOCK), true),
      booleanValue(modelParams.get(Constants.MODEL_USE_MMAP), true),
      splitMode,
      intValue(modelParams.get(Constants.MODEL_MAIN_GPU), 0),
      poolingType,
      attentionType,
      flashAttnType,
      booleanValue(context.get(Constants.CONTEXT_OFFLOAD_KQV), false),
      booleanValue(context.get(Constants.CONTEXT_NO_PERF), false),
      logLevel,
      loraPath
    );
  }

  private Single<java.util.Optional<Path>> resolveLoraPath(ConfigWrapper config) {
    Map<String, Object> modelParams = config.get(Constants.MODEL_PARAMS, Map.of());
    Object repoRaw = modelParams.get(Constants.MODEL_LORA_REPO);
    Object pathRaw = modelParams.get(Constants.MODEL_LORA_REPO_PATH);
    if (repoRaw == null && pathRaw == null) {
      return Single.just(java.util.Optional.empty());
    }
    if (repoRaw == null) {
      return Single.just(java.util.Optional.empty());
    }
    String repo = repoRaw.toString().trim();
    if (repo.isEmpty()) {
      return Single.just(java.util.Optional.empty());
    }
    String path;
    if (repo.startsWith("http://") || repo.startsWith("https://")) {
      LoraSpec spec = parseHuggingFaceUrl(repo);
      if (spec == null) {
        return Single.just(java.util.Optional.empty());
      }
      repo = spec.repo();
      path = spec.file();
    } else {
      if (pathRaw == null) {
        return Single.just(java.util.Optional.empty());
      }
      path = pathRaw.toString().trim();
    }
    if (repo.isEmpty() || path == null || path.isEmpty()) {
      return Single.just(java.util.Optional.empty());
    }

    String modelName = config.get(Constants.MODEL_NAME, UUID.randomUUID().toString());
    return fetchRemoteFile(repo, path, modelName).map(java.util.Optional::of);
  }

  private Single<Path> fetchRemoteFile(String repo, String file, String modelName) {
    FetchModelConfig fetchConfig = new FetchModelConfig(
      repo,
      java.util.List.of(new ModelFile(file, ModelFileType.MODEL)),
      getFileDirectory(modelName)
    );
    return modelFetcher
      .fetchModel(fetchConfig)
      .subscribeOn(RxHelper.blockingScheduler(vertx))
      .observeOn(RxHelper.blockingScheduler(vertx))
      .map(files -> Path.of(files.get(ModelFileType.MODEL)));
  }

  private LoraSpec parseHuggingFaceUrl(String value) {
    try {
      java.net.URI uri = java.net.URI.create(value);
      String host = uri.getHost();
      if (host == null) {
        return null;
      }
      if (!host.endsWith("huggingface.co") && !host.endsWith("hf.co")) {
        return null;
      }
      String path = uri.getPath();
      if (path == null) {
        return null;
      }
      String[] parts = path.split("/");
      int resolveIndex = -1;
      for (int i = 0; i < parts.length; i++) {
        if ("resolve".equals(parts[i])) {
          resolveIndex = i;
          break;
        }
      }
      if (resolveIndex <= 0 || resolveIndex + 2 >= parts.length) {
        return null;
      }
      String repo = String.join("/", java.util.Arrays.copyOfRange(parts, 1, resolveIndex));
      String file = String.join("/", java.util.Arrays.copyOfRange(parts, resolveIndex + 2, parts.length));
      if (repo.isBlank() || file.isBlank()) {
        return null;
      }
      return new LoraSpec(repo, file);
    } catch (Exception e) {
      return null;
    }
  }

  private record LoraSpec(String repo, String file) {}

  private Path resolveModelPath(String modelPath) {
    Path path = Path.of(modelPath);
    return path.isAbsolute() ? path : baseModelPath.resolve(path);
  }

  private Path getFileDirectory(String modelName) {
    Path directory = baseModelPath.resolve(modelName);
    try {
      return Files.createDirectories(directory);
    } catch (FileAlreadyExistsException faee) {
      return directory;
    } catch (Exception e) {
      return createTempDirectory(modelName, e);
    }
  }

  private Path createTempDirectory(String modelName, Exception e) {
    try {
      return Files.createTempDirectory(modelName.replace("/", "-"));
    } catch (IOException ex) {
      throw new RuntimeException(ex);
    }
  }

  private <T extends Enum<T>> T enumValue(Class<T> type, Object value, T defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    String name = value.toString().trim();
    if (name.isEmpty()) {
      return defaultValue;
    }
    try {
      return Enum.valueOf(type, name.toUpperCase());
    } catch (IllegalArgumentException e) {
      return defaultValue;
    }
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

  private int intValue(Object value, int defaultValue) {
    if (value == null) {
      return defaultValue;
    }
    if (value instanceof Number number) {
      return number.intValue();
    }
    return Integer.parseInt(value.toString());
  }
}
