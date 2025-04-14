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

import static io.gravitee.inference.api.service.InferenceAction.CREATE;
import static io.gravitee.inference.api.service.InferenceAction.INFER;
import static org.mockito.Mockito.*;

import io.gravitee.inference.api.InferenceModel;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.service.handler.action.CreateModelAction;
import io.reactivex.rxjava3.core.Observable;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.EventBus;
import io.vertx.rxjava3.core.eventbus.Message;
import io.vertx.rxjava3.core.eventbus.MessageConsumer;
import java.util.HashMap;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@ExtendWith(MockitoExtension.class)
public class InferenceCrudHandlerTest {

  @Mock
  private Vertx vertx;

  @Mock
  private EventBus eventBus;

  @Mock
  private CreateModelAction createModelAction;

  @Mock
  private MessageConsumer<Buffer> messageConsumer;

  @Mock
  private Message<Buffer> message;

  private InferenceCrudHandler inferenceCrudHandler;

  @BeforeEach
  public void setUp() {
    when(vertx.eventBus()).thenReturn(eventBus);
    when(eventBus.<Buffer>consumer(anyString())).thenReturn(messageConsumer);
    when(messageConsumer.toObservable()).thenReturn(Observable.just(message));

    inferenceCrudHandler = new InferenceCrudHandler(vertx, createModelAction);
  }

  @Test
  public void must_handle_create_model_action() {
    InferenceRequest request = new InferenceRequest(CREATE, new HashMap<>());
    when(message.body()).thenReturn(Json.encodeToBuffer(request));
    when(createModelAction.handle(request)).thenReturn(mock(InferenceModel.class));

    inferenceCrudHandler.handle(message);

    verify(message, times(1)).reply(any());
  }

  @Test
  public void must_handle_unsupported_format() {
    InferenceRequest request = new InferenceRequest(INFER, new HashMap<>());
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    inferenceCrudHandler.handle(message);

    verify(message).fail(405, "Unsupported action: INFER");
  }

  @Test
  public void must_handle_null() {
    InferenceRequest request = new InferenceRequest(null, new HashMap<>());
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    inferenceCrudHandler.handle(message);

    verify(message).fail(405, "Unsupported action: null");
  }

  @Test
  public void must_handle_unknown_exception() {
    when(message.body()).thenThrow(new RuntimeException("Test exception"));

    inferenceCrudHandler.handle(message);

    verify(message).fail(400, "Test exception");
  }

  @AfterEach
  public void tearDown() {
    inferenceCrudHandler.close();
  }
}
