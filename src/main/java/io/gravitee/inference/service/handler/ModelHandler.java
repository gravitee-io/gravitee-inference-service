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
package io.gravitee.inference.service.handler;

import static io.gravitee.inference.api.Constants.*;
import static java.util.Optional.ofNullable;

import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.service.repository.ModelRepository;
import io.gravitee.reactive.webclient.api.FetchModelConfig;
import io.gravitee.reactive.webclient.api.ModelFetcher;
import io.gravitee.reactive.webclient.api.ModelFile;
import io.gravitee.reactive.webclient.api.ModelFileType;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.Handler;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.RxHelper;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.Message;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class ModelHandler implements Handler<Message<Buffer>> {

  private static final String GRAVITEE_HOME = "gravitee.home";
  private static final String GRAVITEE_HOME_PATH = System.getProperty(GRAVITEE_HOME);
  private static final String MODEL_NAME = "modelName";

  private final Logger log = LoggerFactory.getLogger(ModelHandler.class);

  private final Vertx vertx;
  private final ModelRepository repository;

  private final Map<String, InferenceHandler> inferenceHandlers = new ConcurrentHashMap<>();
  private final ModelFetcher fetcher;

  public ModelHandler(Vertx vertx, ModelRepository repository, ModelFetcher fetcher) {
    this.vertx = vertx;
    this.repository = repository;
    this.fetcher = fetcher;
  }

  @Override
  public void handle(Message<Buffer> message) {
    try {
      var inferenceRequest = Json.decodeValue(message.body(), InferenceRequest.class);
      switch (inferenceRequest.action()) {
        case START -> handleStart(message, inferenceRequest);
        case STOP -> handleStop(message, inferenceRequest);
        case null, default -> message.fail(405, "Unsupported action: " + inferenceRequest.action());
      }
    } catch (Exception e) {
      message.fail(400, e.getMessage());
    }
  }

  private void handleStart(Message<Buffer> message, InferenceRequest inferenceRequest) {
    var address = String.format(SERVICE_INFERENCE_MODELS_INFER_TEMPLATE, UUID.randomUUID());
    InferenceHandler handler = new InferenceHandler(address, vertx);
    inferenceHandlers.put(address, handler);

    this.fetcher.fetchModel(getModelFetch(new ConfigWrapper(inferenceRequest.payload())))
      .subscribeOn(RxHelper.blockingScheduler(vertx))
      .observeOn(RxHelper.blockingScheduler(vertx))
      .map(modelFiles -> {
        var payload = new HashMap<>(inferenceRequest.payload());
        payload.remove(MODEL_NAME);
        payload.put(MODEL_PATH, modelFiles.get(ModelFileType.MODEL));
        payload.put(TOKENIZER_PATH, modelFiles.get(ModelFileType.TOKENIZER));
        payload.put(CONFIG_JSON_PATH, modelFiles.get(ModelFileType.CONFIG));
        return repository.add(payload);
      })
      .subscribe(
        handler::setModel,
        error -> {
          log.error("Failed to start inference handler", error);
          inferenceHandlers.remove(address);
        }
      );

    message.reply(Buffer.buffer(address));
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

  private void handleStop(Message<Buffer> message, InferenceRequest inferenceRequest) {
    var config = new ConfigWrapper(inferenceRequest.payload());
    var address = config.<String>get(MODEL_ADDRESS_KEY);

    if (inferenceHandlers.containsKey(address)) {
      var inferenceHandler = inferenceHandlers.remove(address);
      inferenceHandler.close();
      repository.remove(inferenceHandler.getModel());
      message.reply(Buffer.buffer(address));
    } else {
      throw new IllegalArgumentException("Could not find inference handler for address: " + address);
    }
  }

  public void close() {
    inferenceHandlers.forEach((__, handler) -> handler.close());
  }

  private Path getFileDirectory(String modelName) {
    Path directory = Path.of(GRAVITEE_HOME_PATH + "/models/" + modelName);
    try {
      return Files.createDirectories(directory);
    } catch (FileAlreadyExistsException faee) {
      log.debug("{} already exists, skip directory creation", directory);
      return directory;
    } catch (Exception e) {
      log.warn("Failed to create directory, creating temp directory", e);
      return createTempDirectory(modelName, e);
    }
  }

  private Path createTempDirectory(String modelName, Exception e) {
    try {
      return Files.createTempDirectory(modelName.replace("/", "-"));
    } catch (IOException ex) {
      log.error("Failed to create temp directory, {}", e.getMessage());
      throw new RuntimeException(ex);
    }
  }
}
