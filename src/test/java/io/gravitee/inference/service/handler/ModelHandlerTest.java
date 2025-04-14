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
import static org.mockito.Mockito.*;

import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.InferenceModel;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.service.repository.Model;
import io.gravitee.inference.service.repository.ModelRepository;
import io.reactivex.rxjava3.core.Observable;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.EventBus;
import io.vertx.rxjava3.core.eventbus.Message;
import io.vertx.rxjava3.core.eventbus.MessageConsumer;
import java.util.HashMap;
import java.util.Map;
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
  private Vertx vertx;

  @Mock
  private EventBus eventBus;

  @Mock
  private ModelRepository repository;

  @Mock
  private MessageConsumer<Buffer> messageConsumer;

  @Mock
  private Message<Buffer> message;

  private ModelHandler modelHandler;

  @BeforeEach
  public void setUp() {
    when(vertx.eventBus()).thenReturn(eventBus);
    when(eventBus.<Buffer>consumer(anyString())).thenReturn(messageConsumer);
    when(messageConsumer.toObservable()).thenReturn(Observable.just(message));

    modelHandler = new ModelHandler(vertx, repository);
  }

  @Test
  public void must_handle_repository_model_start() {
    InferenceRequest request = new InferenceRequest(START, new HashMap<>());
    when(message.body()).thenReturn(Json.encodeToBuffer(request));
    when(repository.add(request)).thenReturn(new Model(0, mock(InferenceModel.class)));

    modelHandler.handle(message);

    verify(message, times(1)).reply(any());
  }

  @Test
  public void must_handle_create_model_action() {
    InferenceRequest request = new InferenceRequest(START, new HashMap<>());
    when(message.body()).thenReturn(Json.encodeToBuffer(request));
    Model model = new Model(0, mock(InferenceModel.class));
    when(repository.add(request)).thenReturn(model);

    modelHandler.handle(message);

    verify(message, times(1)).reply(any());

    ArgumentCaptor<Buffer> captor = ArgumentCaptor.forClass(Buffer.class);
    verify(message).reply(captor.capture());
    Buffer addressBuffer = captor.getValue();

    doNothing().when(repository).remove(eq(model));
    when(message.body())
      .thenReturn(Json.encodeToBuffer(new InferenceRequest(STOP, Map.of(MODEL_ADDRESS_KEY, addressBuffer.toString()))));
    modelHandler.handle(message);

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
