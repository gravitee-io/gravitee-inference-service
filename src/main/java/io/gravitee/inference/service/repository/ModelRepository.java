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
package io.gravitee.inference.service.repository;

import static io.gravitee.inference.api.Constants.*;
import static io.gravitee.inference.api.service.InferenceFormat.ONNX_BERT;
import static io.gravitee.inference.api.service.InferenceType.CLASSIFIER;
import static java.lang.Thread.currentThread;
import static java.util.Optional.ofNullable;

import io.gravitee.inference.api.InferenceModel;
import io.gravitee.inference.api.classifier.ClassifierMode;
import io.gravitee.inference.api.embedding.PoolingMode;
import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.math.vanilla.NativeMath;
import io.gravitee.inference.onnx.bert.classifier.OnnxBertClassifierModel;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import io.gravitee.inference.onnx.bert.embedding.OnnxBertEmbeddingModel;
import io.gravitee.inference.onnx.bert.resource.OnnxBertResource;
import io.gravitee.inference.rest.openai.embedding.OpenAIEmbeddingConfig;
import io.gravitee.inference.rest.openai.embedding.OpenaiEmbeddingInference;
import io.vertx.rxjava3.core.Vertx;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class ModelRepository implements Repository<Model<?>> {

  public static final String OPENAI_API_KEY = "apiKey";
  private static final Logger LOGGER = LoggerFactory.getLogger(ModelRepository.class);
  public static final String OPENAI_DIMENSIONS = "dimensions";
  public static final String OPENAI_MODEL = "model";
  public static final String OPENAI_PROJECT_ID = "projectId";
  public static final String OPENAI_ORGANIZATION_ID = "organizationId";
  public static final String OPENAI_URI = "uri";
  private final Vertx vertx;
  private final Map<Integer, Model<?>> models = new ConcurrentHashMap<>();
  private final Map<Integer, AtomicInteger> counters = new ConcurrentHashMap<>();

  public ModelRepository(Vertx vertx) {
    this.vertx = vertx;
  }

  @Override
  public Model<?> add(Map<String, Object> payload) {
    Integer key = payload.hashCode();

    models.compute(
      key,
      (k, v) -> {
        if (v != null) {
          LOGGER.debug("Model already exists, returning existing model");
          counters.computeIfPresent(
            k,
            (__, cv) -> {
              cv.incrementAndGet();
              return cv;
            }
          );
          return v;
        }

        LOGGER.debug("Model does not exist, creating model");

        var config = new ConfigWrapper(payload);

        var type = InferenceType.valueOf(config.get(INFERENCE_TYPE, CLASSIFIER.name()));
        var format = InferenceFormat.valueOf(config.get(INFERENCE_FORMAT, ONNX_BERT.name()));

        counters.put(k, new AtomicInteger(1));

        return new Model(k, getInferenceModel(type, format, config));
      }
    );

    return models.get(key);
  }

  public int getModelsSize() {
    return models.size();
  }

  public int getModelUsage(int key) {
    return counters.containsKey(key) ? counters.get(key).get() : 0;
  }

  @Override
  public void remove(Model<?> model) {
    counters.computeIfPresent(
      model.key(),
      (k, v) -> {
        var counter = v.decrementAndGet();
        if (counter == 0) {
          LOGGER.debug("Model not in use anymore, tearing down model");
          model.inferenceModel().close();
          models.remove(k);
          LOGGER.debug("Model successfully removed");
          return null;
        }
        LOGGER.debug("Model still in use [{} time(s)]", counter);
        return v;
      }
    );
  }

  private InferenceModel<?, ?, ?> getInferenceModel(InferenceType type, InferenceFormat format, ConfigWrapper config) {
    return switch (type) {
      case CLASSIFIER -> switch (format) {
        case ONNX_BERT -> createInferenceModel(config, this::buildOnnxBertClassifier);
        case null -> throw new IllegalArgumentException("Inference format cannot be null for CLASSIFIER");
        default -> throw new IllegalArgumentException(
          String.format("Unsupported inference format '%s' for type CLASSIFIER. Supported formats: [ONNX_BERT]", format)
        );
      };
      case EMBEDDING -> switch (format) {
        case ONNX_BERT -> createInferenceModel(config, this::buildOnnxBertEmbedding);
        case OPENAI -> createOpenAIEmbeddingInference(config);
        case null -> throw new IllegalArgumentException("Inference format cannot be null for EMBEDDING");
        default -> throw new IllegalArgumentException(
          String.format(
            "Unsupported inference format '%s' for type EMBEDDING. Supported formats: [ONNX_BERT, OPENAI]",
            format
          )
        );
      };
    };
  }

  private OnnxBertClassifierModel buildOnnxBertClassifier(ConfigWrapper config) {
    ClassifierMode mode = ClassifierMode.valueOf(config.get(CLASSIFIER_MODE));
    return new OnnxBertClassifierModel(
      new OnnxBertConfig(
        getResource(config),
        NativeMath.INSTANCE,
        Map.of(CLASSIFIER_MODE, mode, CLASSIFIER_LABELS, ofNullable(config.get(CLASSIFIER_LABELS)).orElse(List.of()))
      )
    );
  }

  private OnnxBertEmbeddingModel buildOnnxBertEmbedding(ConfigWrapper config) {
    PoolingMode poolingMode = PoolingMode.valueOf(config.get(POOLING_MODE, PoolingMode.MEAN.name()));
    return new OnnxBertEmbeddingModel(
      new OnnxBertConfig(
        getResource(config),
        NativeMath.INSTANCE,
        Map.of(
          POOLING_MODE,
          poolingMode,
          MAX_SEQUENCE_LENGTH,
          config.get(MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH_DEFAULT_VALUE)
        )
      )
    );
  }

  private static OnnxBertResource getResource(ConfigWrapper config) {
    return new OnnxBertResource(
      Paths.get(config.<String>get(MODEL_PATH)),
      Paths.get(config.<String>get(TOKENIZER_PATH)),
      ofNullable(config.<String>get(CONFIG_JSON_PATH)).map(Paths::get).orElse(null)
    );
  }

  private <T> T createInferenceModel(ConfigWrapper config, Function<ConfigWrapper, T> function) {
    var currentClassLoader = currentThread().getContextClassLoader();
    try {
      currentThread().setContextClassLoader(ModelRepository.class.getClassLoader());
      return function.apply(config);
    } finally {
      currentThread().setContextClassLoader(currentClassLoader);
    }
  }

  private OpenaiEmbeddingInference createOpenAIEmbeddingInference(ConfigWrapper config) {
    OpenAIEmbeddingConfig openAIEmbeddingConfig = new OpenAIEmbeddingConfig(
      config.get(OPENAI_URI),
      config.get(OPENAI_API_KEY),
      config.get(OPENAI_ORGANIZATION_ID),
      config.get(OPENAI_PROJECT_ID),
      config.get(OPENAI_MODEL),
      config.get(OPENAI_DIMENSIONS)
    );
    var inferenceService = new OpenaiEmbeddingInference(openAIEmbeddingConfig, vertx);
    LOGGER.debug("OpenAI Embedding inference service started {} with config {}", inferenceService, openAIEmbeddingConfig);
    return inferenceService;
  }
}
