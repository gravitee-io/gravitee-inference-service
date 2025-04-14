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
package io.gravitee.inference.service.handler.action;

import static io.gravitee.inference.api.Constants.*;
import static java.lang.Thread.currentThread;

import io.gravitee.inference.api.InferenceModel;
import io.gravitee.inference.api.classifier.ClassifierMode;
import io.gravitee.inference.api.embedding.PoolingMode;
import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.math.vanilla.NativeMath;
import io.gravitee.inference.onnx.bert.classifier.OnnxBertClassifierModel;
import io.gravitee.inference.onnx.bert.config.OnnxBertConfig;
import io.gravitee.inference.onnx.bert.embedding.OnnxBertEmbeddingModel;
import io.gravitee.inference.onnx.bert.resource.OnnxBertResource;
import java.nio.file.Paths;
import java.util.Map;
import java.util.function.Function;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class CreateModelAction implements ActionHandler<InferenceModel<?, ?, ?>> {

  @Override
  public InferenceModel<?, String, ?> handle(InferenceRequest request) {
    var config = new ConfigWrapper(request.payload());
    var type = InferenceType.valueOf(config.get(INFERENCE_TYPE, InferenceType.CLASSIFIER.name()));
    var format = InferenceFormat.valueOf(config.get(INFERENCE_FORMAT, InferenceFormat.ONNX_BERT.name()));
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
        Map.of(CLASSIFIER_MODE, mode, CLASSIFIER_LABELS, config.get(CLASSIFIER_LABELS))
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
    return new OnnxBertResource(Paths.get(config.<String>get(MODEL_PATH)), Paths.get(config.<String>get(TOKENIZER_PATH)));
  }

  // This is to access native libraries present in the classpath
  private <T> T createInferenceModel(ConfigWrapper config, Function<ConfigWrapper, T> function) {
    var currentClassLoader = currentThread().getContextClassLoader();
    try {
      currentThread().setContextClassLoader(CreateModelAction.class.getClassLoader());
      return function.apply(config);
    } finally {
      currentThread().setContextClassLoader(currentClassLoader);
    }
  }
}
