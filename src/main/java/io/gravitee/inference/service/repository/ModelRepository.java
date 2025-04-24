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

import io.gravitee.inference.api.classifier.ClassifierMode;
import io.gravitee.inference.api.embedding.PoolingMode;
import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.math.vanilla.NativeMath;
import io.gravitee.inference.onnx.bert.OnnxBertInference;
import io.gravitee.inference.onnx.bert.classifier.OnnxBertClassifierModel;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import io.gravitee.inference.onnx.bert.embedding.OnnxBertEmbeddingModel;
import io.gravitee.inference.onnx.bert.resource.OnnxBertResource;
import java.nio.file.Path;
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
public class ModelRepository implements Repository<Model> {

  private static final Logger LOGGER = LoggerFactory.getLogger(ModelRepository.class);
  private final Map<Integer, Model> models = new ConcurrentHashMap<>();
  private final Map<Integer, AtomicInteger> counters = new ConcurrentHashMap<>();

  @Override
  public Model add(InferenceRequest request) {
    Integer key = request.payload().hashCode();

    if (models.containsKey(key)) {
      LOGGER.debug("Model already exists, returning existing model");
      counters.computeIfPresent(
        key,
        (k, v) -> {
          v.incrementAndGet();
          return v;
        }
      );
      return models.get(key);
    }

    LOGGER.debug("Model does not exist, creating model");

    var config = new ConfigWrapper(request.payload());
    var type = InferenceType.valueOf(config.get(INFERENCE_TYPE, CLASSIFIER.name()));
    var format = InferenceFormat.valueOf(config.get(INFERENCE_FORMAT, ONNX_BERT.name()));

    var model = new Model(key, getInferenceModel(type, format, config));

    models.put(key, model);
    counters.put(key, new AtomicInteger(1));

    return model;
  }

  public int getModelsSize() {
    return models.size();
  }

  public int getModelUsage(int key) {
    return counters.containsKey(key) ? counters.get(key).get() : 0;
  }

  @Override
  public void remove(Model model) {
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

  private OnnxBertInference<? extends Record> getInferenceModel(
    InferenceType type,
    InferenceFormat format,
    ConfigWrapper config
  ) {
    return switch (type) {
      case CLASSIFIER -> switch (format) {
        case ONNX_BERT -> createInferenceModel(config, this::buildOnnxBertClassifier);
      };
      case EMBEDDING -> switch (format) {
        case ONNX_BERT -> createInferenceModel(config, this::buildOnnxBertEmbedding);
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

  // This is to access native libraries present in the classpath
  private <T> T createInferenceModel(ConfigWrapper config, Function<ConfigWrapper, T> function) {
    var currentClassLoader = currentThread().getContextClassLoader();
    try {
      currentThread().setContextClassLoader(ModelRepository.class.getClassLoader());
      return function.apply(config);
    } finally {
      currentThread().setContextClassLoader(currentClassLoader);
    }
  }
}
