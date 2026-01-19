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
package io.gravitee.inference.service.model;

import static io.gravitee.inference.api.Constants.*;
import static java.lang.Thread.currentThread;
import static java.util.Optional.ofNullable;

import io.gravitee.inference.api.InferenceModel;
import io.gravitee.inference.api.classifier.ClassifierMode;
import io.gravitee.inference.api.embedding.PoolingMode;
import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.math.vanilla.NativeMath;
import io.gravitee.inference.onnx.OnnxInference;
import io.gravitee.inference.onnx.bert.classifier.OnnxBertClassifierModel;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import io.gravitee.inference.onnx.bert.embedding.OnnxBertEmbeddingModel;
import io.gravitee.inference.onnx.bert.resource.OnnxBertResource;
import io.gravitee.inference.service.repository.HandlerRepository;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class OnnxModelFactory implements InferenceModelFactory<OnnxInference<?, ?, ?>> {

  private static final Logger LOGGER = LoggerFactory.getLogger(OnnxModelFactory.class);
  private static final String EXCEPTION_TEMPLATE = "Unsupported inference format '%s'";

  public OnnxInference<?, ?, ?> build(ConfigWrapper config) {
    InferenceType type = InferenceType.valueOf(config.get(INFERENCE_TYPE));
    InferenceFormat format = InferenceFormat.valueOf(config.get(INFERENCE_FORMAT));

    return switch (format) {
      case ONNX_BERT -> switch (type) {
        case CLASSIFIER -> createInferenceModel(config, this::buildOnnxBertClassifier);
        case EMBEDDING -> createInferenceModel(config, this::buildOnnxBertEmbedding);
        default -> throw new IllegalArgumentException(
          String.format("Unsupported inference type '%s' for format '%s'", type, format)
        );
      };
      default -> throw new IllegalArgumentException(String.format(EXCEPTION_TEMPLATE, format));
    };
  }

  private OnnxBertClassifierModel buildOnnxBertClassifier(ConfigWrapper config) {
    ClassifierMode mode = ClassifierMode.valueOf(config.get(CLASSIFIER_MODE));
    return new OnnxBertClassifierModel(
      new OnnxBertConfig(
        getResource(config),
        NativeMath.INSTANCE,
        Map.of(
          CLASSIFIER_MODE,
          mode,
          CLASSIFIER_LABELS,
          ofNullable(config.get(CLASSIFIER_LABELS)).orElse(List.of()),
          DISCARDED_LABELS,
          ofNullable(config.get(DISCARDED_LABELS)).orElse(List.of())
        )
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
    LOGGER.debug("Getting resource from config: {}", config);
    return new OnnxBertResource(
      Paths.get(config.<String>get(MODEL_PATH)),
      Paths.get(config.<String>get(TOKENIZER_PATH)),
      ofNullable(config.<String>get(CONFIG_JSON_PATH)).map(Paths::get).orElse(null)
    );
  }

  private <T extends InferenceModel<?, ?, ?>> T createInferenceModel(
    ConfigWrapper config,
    Function<ConfigWrapper, T> function
  ) {
    var currentClassLoader = currentThread().getContextClassLoader();
    try {
      currentThread().setContextClassLoader(HandlerRepository.class.getClassLoader());
      return function.apply(config);
    } finally {
      currentThread().setContextClassLoader(currentClassLoader);
    }
  }
}
