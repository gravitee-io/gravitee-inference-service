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

import static io.gravitee.inference.api.Constants.MODEL_ADDRESS_KEY;
import static io.gravitee.inference.api.service.InferenceAction.*;
import static io.reactivex.rxjava3.core.Observable.fromRunnable;
import static io.reactivex.rxjava3.core.Observable.timer;
import static org.mockito.Mockito.*;

import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.InferenceModel;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.service.repository.Model;
import io.gravitee.inference.service.repository.ModelRepository;
import io.gravitee.reactive.webclient.api.ModelFetcher;
import io.gravitee.reactive.webclient.api.ModelFile;
import io.gravitee.reactive.webclient.api.ModelFileType;
import io.reactivex.rxjava3.core.Observable;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.EventBus;
import io.vertx.rxjava3.core.eventbus.Message;
import io.vertx.rxjava3.core.eventbus.MessageConsumer;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@ExtendWith(MockitoExtension.class)
public class ModelHandlerTest {

  @Mock
  private EventBus eventBus;

  @Mock
  private Vertx vertx;

  @Mock
  private ModelFetcher fetcher;

  @Mock
  private ModelRepository repository;

  @Mock
  private MessageConsumer<Buffer> messageConsumer;

  @Mock
  private Message<Buffer> message;

  private ModelHandler modelHandler;

  private io.vertx.core.Vertx delegate;

  @BeforeEach
  public void setUp() {
    delegate = io.vertx.core.Vertx.vertx();
    lenient().when(vertx.getDelegate()).thenReturn(delegate);
    lenient().when(vertx.eventBus()).thenReturn(eventBus);

    lenient().when(eventBus.<Buffer>consumer(anyString())).thenReturn(messageConsumer);
    Observable<Message<Buffer>> observable = Observable.just(message);

    lenient().when(messageConsumer.toObservable()).thenReturn(observable);
    lenient().when(message.body()).thenReturn(Buffer.buffer("some_address"));
    lenient()
      .when(fetcher.fetchModel(any()))
      .thenReturn(
        Single.just(
          Map.of(
            ModelFileType.CONFIG,
            "/path/to/config.json",
            ModelFileType.TOKENIZER,
            "/path/to/tokenizer.json",
            ModelFileType.MODEL,
            "/path/to/model.json"
          )
        )
      );

    modelHandler = new ModelHandler(vertx, repository, fetcher);
  }

  @Test
  public void must_handle_create_model_action() {
    InferenceRequest request = new InferenceRequest(START, new HashMap<>());
    when(message.body()).thenReturn(Json.encodeToBuffer(request));
    Model model = new Model(0, mock(InferenceModel.class));
    when(repository.add(any())).thenReturn(model);

    fromRunnable(() -> modelHandler.handle(message))
      .flatMap(__ -> timer(2, TimeUnit.SECONDS))
      .test()
      .awaitDone(3, TimeUnit.SECONDS)
      .assertComplete()
      .assertNoErrors();

    ArgumentCaptor<Buffer> captor = ArgumentCaptor.forClass(Buffer.class);
    verify(message, times(1)).reply(captor.capture());
    Buffer addressBuffer = captor.getValue();

    doNothing().when(repository).remove(eq(model));
    when(message.body())
      .thenReturn(Json.encodeToBuffer(new InferenceRequest(STOP, Map.of(MODEL_ADDRESS_KEY, addressBuffer.toString()))));

    fromRunnable(() -> modelHandler.handle(message))
      .flatMap(__ -> timer(2, TimeUnit.SECONDS))
      .test()
      .awaitDone(3, TimeUnit.SECONDS)
      .assertComplete()
      .assertNoErrors();

    verify(message, times(2)).reply(any());
  }

  @Test
  public void must_handle_stop_model_action() {
    InferenceRequest request = new InferenceRequest(STOP, Map.of(MODEL_ADDRESS_KEY, "unknownAddress"));
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    modelHandler.handle(message);

    verify(message).fail(400, "Could not find inference handler for address: unknownAddress");
  }

  @Test
  public void must_handle_unsupported_format() {
    InferenceRequest request = new InferenceRequest(INFER, new HashMap<>());
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    modelHandler.handle(message);

    verify(message).fail(405, "Unsupported action: INFER");
  }

  @Test
  public void must_handle_null() {
    InferenceRequest request = new InferenceRequest(null, new HashMap<>());
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    modelHandler.handle(message);

    verify(message).fail(405, "Unsupported action: null");
  }

  @Test
  public void must_handle_unknown_exception() {
    when(message.body()).thenThrow(new RuntimeException("Test exception"));

    modelHandler.handle(message);

    verify(message).fail(400, "Test exception");
  }

  @AfterEach
  public void tearDown() {
    modelHandler.close();
  }
}
