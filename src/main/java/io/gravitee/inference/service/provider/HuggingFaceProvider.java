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

import static io.gravitee.inference.api.Constants.*;
import static java.util.Optional.ofNullable;

import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.service.repository.Model;
import io.gravitee.inference.service.repository.ModelRepository;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HuggingFaceProvider implements ModelProvider {

  public static final String MODEL_NAME = "modelName";
  private final Logger LOGGER = LoggerFactory.getLogger(HuggingFaceProvider.class);

  private final Vertx vertx;
  private final String modelPath;
  private final ModelFetcher modelFetcher;

  public HuggingFaceProvider(Vertx vertx, String modelPath) {
    this.vertx = vertx;
    this.modelPath = modelPath;
    this.modelFetcher = new HuggingFaceDownloader(vertx);
  }

  @Override
  public Single<Model<?>> loadModel(InferenceRequest inferenceRequest, ModelRepository repository) {
    return fetchModelFiles(inferenceRequest)
      .map(modelFiles -> createModelPayload(inferenceRequest.payload(), modelFiles))
      .map(this::addInferenceInfo)
      .map(repository::add);
  }

  public Map<String, Object> addInferenceInfo(Map<String, Object> inferenceRequest) {
    inferenceRequest.put(INFERENCE_TYPE, InferenceType.EMBEDDING.name());
    inferenceRequest.put(INFERENCE_FORMAT, InferenceFormat.ONNX_BERT.name());

    return inferenceRequest;
  }

  private Single<Map<ModelFileType, String>> fetchModelFiles(InferenceRequest request) {
    ConfigWrapper config = new ConfigWrapper(request.payload());
    return modelFetcher
      .fetchModel(getModelFetch(config))
      .subscribeOn(RxHelper.blockingScheduler(vertx))
      .observeOn(RxHelper.blockingScheduler(vertx));
  }

  private Map<String, Object> createModelPayload(
    Map<String, Object> originalPayload,
    Map<ModelFileType, String> modelFiles
  ) {
    var payload = new HashMap<>(originalPayload);
    payload.remove(MODEL_NAME);
    payload.put(MODEL_PATH, modelFiles.get(ModelFileType.MODEL));
    payload.put(TOKENIZER_PATH, modelFiles.get(ModelFileType.TOKENIZER));
    payload.put(CONFIG_JSON_PATH, modelFiles.get(ModelFileType.CONFIG));
    return payload;
  }

  private FetchModelConfig getModelFetch(ConfigWrapper payload) {
    String modelName = payload.get(MODEL_NAME, UUID.randomUUID().toString());
    var fileList = new ArrayList<ModelFile>();
    ofNullable(payload.<String>get(MODEL_PATH)).ifPresent(path -> fileList.add(new ModelFile(path, ModelFileType.MODEL)));
    ofNullable(payload.<String>get(TOKENIZER_PATH))
      .ifPresent(path -> fileList.add(new ModelFile(path, ModelFileType.TOKENIZER)));
    ofNullable(payload.<String>get(CONFIG_JSON_PATH))
      .ifPresent(path -> fileList.add(new ModelFile(path, ModelFileType.CONFIG)));
    return new FetchModelConfig(modelName, fileList, getFileDirectory(modelName));
  }

  private Path getFileDirectory(String modelName) {
    Path directory = Path.of(this.modelPath + "/" + modelName);
    try {
      return Files.createDirectories(directory);
    } catch (FileAlreadyExistsException faee) {
      LOGGER.debug("{} already exists, skip directory creation", directory);
      return directory;
    } catch (Exception e) {
      LOGGER.warn("Failed to create directory, creating temp directory", e);
      return createTempDirectory(modelName, e);
    }
  }

  private Path createTempDirectory(String modelName, Exception e) {
    try {
      return Files.createTempDirectory(modelName.replace("/", "-"));
    } catch (IOException ex) {
      LOGGER.error("Failed to create temp directory, {}", e.getMessage());
      throw new RuntimeException(ex);
    }
  }
}
