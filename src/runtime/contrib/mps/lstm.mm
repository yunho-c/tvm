/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/ffi/reflection/registry.h>

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mps_utils.h"

@interface TVMMPSDenseDataSource : NSObject <MPSCNNConvolutionDataSource> {
 @private
  MPSCNNConvolutionDescriptor* desc_;
  float* weights_;
  float* bias_;
  NSUInteger weight_bytes_;
  NSUInteger bias_bytes_;
  NSString* label_;
}

- (instancetype)initWithInputChannels:(NSUInteger)input_channels
                      outputChannels:(NSUInteger)output_channels
                          neuronType:(MPSCNNNeuronType)neuron_type
                              weight:(const float*)weight
                         weightBytes:(NSUInteger)weight_bytes
                                bias:(const float*)bias
                           biasBytes:(NSUInteger)bias_bytes
                               label:(NSString*)label;
@end

@implementation TVMMPSDenseDataSource
- (instancetype)initWithInputChannels:(NSUInteger)input_channels
                      outputChannels:(NSUInteger)output_channels
                          neuronType:(MPSCNNNeuronType)neuron_type
                              weight:(const float*)weight
                         weightBytes:(NSUInteger)weight_bytes
                                bias:(const float*)bias
                           biasBytes:(NSUInteger)bias_bytes
                               label:(NSString*)label {
  self = [super init];
  if (self == nil) {
    return nil;
  }

  desc_ = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:1
                                                                  kernelHeight:1
                                                          inputFeatureChannels:input_channels
                                                         outputFeatureChannels:output_channels];
  if (neuron_type != MPSCNNNeuronTypeNone) {
    desc_.fusedNeuronDescriptor = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:neuron_type];
  }

  weight_bytes_ = weight_bytes;
  weights_ = static_cast<float*>(malloc(weight_bytes_));
  ICHECK(weights_ != nullptr);
  memcpy(weights_, weight, weight_bytes_);

  bias_bytes_ = bias_bytes;
  bias_ = nullptr;
  if (bias != nullptr && bias_bytes_ != 0) {
    bias_ = static_cast<float*>(malloc(bias_bytes_));
    ICHECK(bias_ != nullptr);
    memcpy(bias_, bias, bias_bytes_);
  }

  label_ = label;
  return self;
}

- (MPSDataType)dataType {
  return MPSDataTypeFloat32;
}

- (MPSCNNConvolutionDescriptor*)descriptor {
  return desc_;
}

- (void*)weights {
  return weights_;
}

- (float*)biasTerms {
  return bias_;
}

- (BOOL)load {
  return YES;
}

- (void)purge {
}

- (NSString*)label {
  return label_;
}

- (id)copyWithZone:(NSZone*)zone {
  return self;
}

- (void)dealloc {
  if (weights_ != nullptr) {
    free(weights_);
    weights_ = nullptr;
  }
  if (bias_ != nullptr) {
    free(bias_);
    bias_ = nullptr;
  }
  [super dealloc];
}
@end

namespace tvm {
namespace contrib {

using namespace runtime;

namespace {

struct LSTMCacheKey {
  int device_id{0};
  int input_size{0};
  int hidden_size{0};
  uintptr_t w_ih{0};
  uintptr_t w_hh{0};
  uintptr_t b_ih{0};
  uintptr_t b_hh{0};

  bool operator==(const LSTMCacheKey& other) const {
    return device_id == other.device_id && input_size == other.input_size && hidden_size == other.hidden_size &&
           w_ih == other.w_ih && w_hh == other.w_hh && b_ih == other.b_ih && b_hh == other.b_hh;
  }
};

struct LSTMCacheKeyHash {
  std::size_t operator()(const LSTMCacheKey& k) const {
    auto h = std::hash<uintptr_t>{};
    std::size_t out = 0;
    out ^= std::hash<int>{}(k.device_id) + 0x9e3779b9 + (out << 6) + (out >> 2);
    out ^= std::hash<int>{}(k.input_size) + 0x9e3779b9 + (out << 6) + (out >> 2);
    out ^= std::hash<int>{}(k.hidden_size) + 0x9e3779b9 + (out << 6) + (out >> 2);
    out ^= h(k.w_ih) + 0x9e3779b9 + (out << 6) + (out >> 2);
    out ^= h(k.w_hh) + 0x9e3779b9 + (out << 6) + (out >> 2);
    out ^= h(k.b_ih) + 0x9e3779b9 + (out << 6) + (out >> 2);
    out ^= h(k.b_hh) + 0x9e3779b9 + (out << 6) + (out >> 2);
    return out;
  }
};

struct LSTMCacheValue {
  MPSRNNMatrixInferenceLayer* layer{nil};
  NSArray<id<MPSCNNConvolutionDataSource>>* retained_data_sources{nil};
};

static std::mutex g_lstm_mu;
static std::unordered_map<LSTMCacheKey, LSTMCacheValue, LSTMCacheKeyHash> g_lstm_cache;

std::vector<float> CopyTensorToCPUFloat32(MetalThreadEntry* entry_ptr, DLTensor* t) {
  ICHECK(TypeMatch(t->dtype, kDLFloat, 32));
  ICHECK(ffi::IsContiguous(*t));
  ICHECK_EQ(t->device.device_type, kDLMetal);
  ICHECK(t->data != nullptr);

  size_t nbytes = ffi::GetDataSize(*t);
  ICHECK_EQ(nbytes % sizeof(float), 0);

  runtime::metal::MetalThreadEntry* rt = runtime::metal::MetalThreadEntry::ThreadLocal();
  id<MTLBuffer> src = (__bridge id<MTLBuffer>)(t->data);
  id<MTLBuffer> tmp = rt->GetTempBuffer(t->device, nbytes);

  runtime::metal::Stream* stream = entry_ptr->metal_api->CastStreamOrGetDefault(
      entry_ptr->metal_api->GetCurrentStream(t->device), t->device.device_id);
  id<MTLCommandBuffer> cb = stream->GetCommandBuffer("tvm.contrib.mps.lstm.copy_weights");
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromBuffer:src sourceOffset:0 toBuffer:tmp destinationOffset:0 size:nbytes];
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];

  std::vector<float> out(nbytes / sizeof(float));
  memcpy(out.data(), [tmp contents], nbytes);
  return out;
}

std::vector<float> SlicePackedGates(const std::vector<float>& packed, int gate, int hidden_size, int in_size) {
  std::vector<float> out(static_cast<size_t>(hidden_size) * static_cast<size_t>(in_size));
  const size_t gate_row0 = static_cast<size_t>(gate) * static_cast<size_t>(hidden_size);
  for (int r = 0; r < hidden_size; ++r) {
    const size_t packed_row = (gate_row0 + static_cast<size_t>(r)) * static_cast<size_t>(in_size);
    const size_t out_row = static_cast<size_t>(r) * static_cast<size_t>(in_size);
    memcpy(out.data() + out_row, packed.data() + packed_row, static_cast<size_t>(in_size) * sizeof(float));
  }
  return out;
}

std::vector<float> SlicePackedBias(const std::vector<float>& b_ih,
                                  const std::vector<float>& b_hh,
                                  int gate,
                                  int hidden_size) {
  std::vector<float> out(static_cast<size_t>(hidden_size));
  const size_t gate_row0 = static_cast<size_t>(gate) * static_cast<size_t>(hidden_size);
  for (int i = 0; i < hidden_size; ++i) {
    out[static_cast<size_t>(i)] =
        b_ih[gate_row0 + static_cast<size_t>(i)] + b_hh[gate_row0 + static_cast<size_t>(i)];
  }
  return out;
}

LSTMCacheValue GetOrCreateLSTMLayer(MetalThreadEntry* entry_ptr,
                                   id<MTLDevice> dev,
                                   int input_size,
                                   int hidden_size,
                                   DLTensor* w_ih,
                                   DLTensor* w_hh,
                                   DLTensor* b_ih,
                                   DLTensor* b_hh) {
  LSTMCacheKey key;
  key.device_id = w_ih->device.device_id;
  key.input_size = input_size;
  key.hidden_size = hidden_size;
  key.w_ih = reinterpret_cast<uintptr_t>(w_ih->data);
  key.w_hh = reinterpret_cast<uintptr_t>(w_hh->data);
  key.b_ih = reinterpret_cast<uintptr_t>(b_ih->data);
  key.b_hh = reinterpret_cast<uintptr_t>(b_hh->data);

  {
    std::lock_guard<std::mutex> lock(g_lstm_mu);
    auto it = g_lstm_cache.find(key);
    if (it != g_lstm_cache.end()) {
      return it->second;
    }
  }

  std::vector<float> w_ih_cpu = CopyTensorToCPUFloat32(entry_ptr, w_ih);
  std::vector<float> w_hh_cpu = CopyTensorToCPUFloat32(entry_ptr, w_hh);
  std::vector<float> b_ih_cpu = CopyTensorToCPUFloat32(entry_ptr, b_ih);
  std::vector<float> b_hh_cpu = CopyTensorToCPUFloat32(entry_ptr, b_hh);

  ICHECK_EQ(static_cast<int>(w_ih->shape[0]), 4 * hidden_size);
  ICHECK_EQ(static_cast<int>(w_ih->shape[1]), input_size);
  ICHECK_EQ(static_cast<int>(w_hh->shape[0]), 4 * hidden_size);
  ICHECK_EQ(static_cast<int>(w_hh->shape[1]), hidden_size);
  ICHECK_EQ(static_cast<int>(b_ih->shape[0]), 4 * hidden_size);
  ICHECK_EQ(static_cast<int>(b_hh->shape[0]), 4 * hidden_size);

  MPSLSTMDescriptor* desc =
      [MPSLSTMDescriptor createLSTMDescriptorWithInputFeatureChannels:static_cast<NSUInteger>(input_size)
                                                outputFeatureChannels:static_cast<NSUInteger>(hidden_size)];
  desc.useFloat32Weights = YES;

  // Gate order is expected to match PyTorch: input, forget, cell, output.
  // If this turns out to be mismatched, swap the slices here.
  const int gate_input = 0;
  const int gate_forget = 1;
  const int gate_cell = 2;
  const int gate_output = 3;

  auto make_input_ds = [&](const char* label, int gate, MPSCNNNeuronType neuron) {
    std::vector<float> w = SlicePackedGates(w_ih_cpu, gate, hidden_size, input_size);
    std::vector<float> b = SlicePackedBias(b_ih_cpu, b_hh_cpu, gate, hidden_size);
    return [[TVMMPSDenseDataSource alloc] initWithInputChannels:static_cast<NSUInteger>(input_size)
                                                outputChannels:static_cast<NSUInteger>(hidden_size)
                                                    neuronType:neuron
                                                        weight:w.data()
                                                   weightBytes:w.size() * sizeof(float)
                                                          bias:b.data()
                                                     biasBytes:b.size() * sizeof(float)
                                                         label:[NSString stringWithUTF8String:label]];
  };

  auto make_recurrent_ds = [&](const char* label, int gate, MPSCNNNeuronType neuron) {
    std::vector<float> w = SlicePackedGates(w_hh_cpu, gate, hidden_size, hidden_size);
    return [[TVMMPSDenseDataSource alloc] initWithInputChannels:static_cast<NSUInteger>(hidden_size)
                                                outputChannels:static_cast<NSUInteger>(hidden_size)
                                                    neuronType:neuron
                                                        weight:w.data()
                                                   weightBytes:w.size() * sizeof(float)
                                                          bias:nullptr
                                                     biasBytes:0
                                                         label:[NSString stringWithUTF8String:label]];
  };

  id<MPSCNNConvolutionDataSource> i_in = make_input_ds("lstm.inputGateInput", gate_input, MPSCNNNeuronTypeSigmoid);
  id<MPSCNNConvolutionDataSource> i_rec =
      make_recurrent_ds("lstm.inputGateRecurrent", gate_input, MPSCNNNeuronTypeNone);
  id<MPSCNNConvolutionDataSource> f_in = make_input_ds("lstm.forgetGateInput", gate_forget, MPSCNNNeuronTypeSigmoid);
  id<MPSCNNConvolutionDataSource> f_rec =
      make_recurrent_ds("lstm.forgetGateRecurrent", gate_forget, MPSCNNNeuronTypeNone);
  id<MPSCNNConvolutionDataSource> c_in = make_input_ds("lstm.cellGateInput", gate_cell, MPSCNNNeuronTypeTanH);
  id<MPSCNNConvolutionDataSource> c_rec = make_recurrent_ds("lstm.cellGateRecurrent", gate_cell, MPSCNNNeuronTypeNone);
  id<MPSCNNConvolutionDataSource> o_in = make_input_ds("lstm.outputGateInput", gate_output, MPSCNNNeuronTypeSigmoid);
  id<MPSCNNConvolutionDataSource> o_rec =
      make_recurrent_ds("lstm.outputGateRecurrent", gate_output, MPSCNNNeuronTypeNone);

  desc.inputGateInputWeights = i_in;
  desc.inputGateRecurrentWeights = i_rec;
  desc.forgetGateInputWeights = f_in;
  desc.forgetGateRecurrentWeights = f_rec;
  desc.cellGateInputWeights = c_in;
  desc.cellGateRecurrentWeights = c_rec;
  desc.outputGateInputWeights = o_in;
  desc.outputGateRecurrentWeights = o_rec;

  MPSRNNMatrixInferenceLayer* layer =
      [[MPSRNNMatrixInferenceLayer alloc] initWithDevice:dev rnnDescriptor:(const MPSRNNDescriptor*)desc];
  ICHECK(layer != nil);
  layer.storeAllIntermediateStates = NO;
  layer.recurrentOutputIsTemporary = NO;

  NSArray<id<MPSCNNConvolutionDataSource>>* retained =
      @[ i_in, i_rec, f_in, f_rec, c_in, c_rec, o_in, o_rec ];

  LSTMCacheValue value;
  value.layer = layer;
  value.retained_data_sources = retained;

  {
    std::lock_guard<std::mutex> lock(g_lstm_mu);
    g_lstm_cache.emplace(key, value);
  }
  return value;
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvm.contrib.mps.lstm", [](ffi::PackedArgs args, ffi::Any* ret) {
    auto x = args[0].cast<DLTensor*>();
    auto weight_ih = args[1].cast<DLTensor*>();
    auto weight_hh = args[2].cast<DLTensor*>();
    auto bias_ih = args[3].cast<DLTensor*>();
    auto bias_hh = args[4].cast<DLTensor*>();
    auto h0 = args[5].cast<DLTensor*>();
    auto c0 = args[6].cast<DLTensor*>();
    auto out = args[7].cast<DLTensor*>();
    auto h_n = args[8].cast<DLTensor*>();
    auto c_n = args[9].cast<DLTensor*>();
    int hidden_size = args[10].cast<int>();
    int num_layers = args[11].cast<int>();
    bool batch_first = args[12].cast<bool>();
    bool bidirectional = args[13].cast<bool>();

    ICHECK_EQ(num_layers, 1) << "tvm.contrib.mps.lstm currently supports num_layers=1";
    ICHECK(!bidirectional) << "tvm.contrib.mps.lstm currently supports bidirectional=false";
    ICHECK(TypeMatch(x->dtype, kDLFloat, 32));
    ICHECK(TypeMatch(weight_ih->dtype, kDLFloat, 32));
    ICHECK(TypeMatch(weight_hh->dtype, kDLFloat, 32));
    ICHECK(TypeMatch(bias_ih->dtype, kDLFloat, 32));
    ICHECK(TypeMatch(bias_hh->dtype, kDLFloat, 32));
    ICHECK(TypeMatch(out->dtype, kDLFloat, 32));
    ICHECK(TypeMatch(h_n->dtype, kDLFloat, 32));
    ICHECK(TypeMatch(c_n->dtype, kDLFloat, 32));

    ICHECK(ffi::IsContiguous(*x));
    ICHECK(ffi::IsContiguous(*weight_ih));
    ICHECK(ffi::IsContiguous(*weight_hh));
    ICHECK(ffi::IsContiguous(*bias_ih));
    ICHECK(ffi::IsContiguous(*bias_hh));
    ICHECK(ffi::IsContiguous(*out));
    ICHECK(ffi::IsContiguous(*h_n));
    ICHECK(ffi::IsContiguous(*c_n));

    ICHECK_EQ(x->device.device_type, kDLMetal);
    ICHECK_EQ(out->device.device_type, kDLMetal);
    ICHECK_EQ(h_n->device.device_type, kDLMetal);
    ICHECK_EQ(c_n->device.device_type, kDLMetal);
    ICHECK_EQ(weight_ih->device.device_type, kDLMetal);
    ICHECK_EQ(weight_hh->device.device_type, kDLMetal);
    ICHECK_EQ(bias_ih->device.device_type, kDLMetal);
    ICHECK_EQ(bias_hh->device.device_type, kDLMetal);

    int seq_len = static_cast<int>(batch_first ? x->shape[1] : x->shape[0]);
    int batch = static_cast<int>(batch_first ? x->shape[0] : x->shape[1]);
    int input_size = static_cast<int>(x->shape[2]);

    ICHECK_EQ(hidden_size * 4, static_cast<int>(weight_ih->shape[0]));
    ICHECK_EQ(input_size, static_cast<int>(weight_ih->shape[1]));

    ICHECK_EQ(out->ndim, 3);
    ICHECK_EQ(out->shape[0], batch_first ? batch : seq_len);
    ICHECK_EQ(out->shape[1], batch_first ? seq_len : batch);
    ICHECK_EQ(out->shape[2], hidden_size);

    // Initial states are not supported yet. The first milestone uses zero init.
    (void)h0;
    (void)c0;

    MetalThreadEntry* entry_ptr = MetalThreadEntry::ThreadLocal();
    id<MTLDevice> dev = entry_ptr->metal_api->GetDevice(x->device);
    runtime::metal::Stream* stream = entry_ptr->metal_api->CastStreamOrGetDefault(
        entry_ptr->metal_api->GetCurrentStream(x->device), x->device.device_id);
    id<MTLCommandBuffer> cb = stream->GetCommandBuffer("tvm.contrib.mps.lstm");

    LSTMCacheValue cached = GetOrCreateLSTMLayer(entry_ptr, dev, input_size, hidden_size,
                                                weight_ih, weight_hh, bias_ih, bias_hh);
    MPSRNNMatrixInferenceLayer* layer = cached.layer;

    // Encode expects a sequence of MPSMatrix objects. Each matrix stores one timestep with vectors as rows.
    const size_t out_state_elems = static_cast<size_t>(batch) * static_cast<size_t>(hidden_size);

    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)(x->data);
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)(out->data);

    NSMutableArray<MPSMatrix*>* src_mats = [NSMutableArray arrayWithCapacity:seq_len];
    NSMutableArray<MPSMatrix*>* dst_mats = [NSMutableArray arrayWithCapacity:seq_len];

    for (int t = 0; t < seq_len; ++t) {
      const NSUInteger x_off = batch_first ? static_cast<NSUInteger>(t) * static_cast<NSUInteger>(input_size * sizeof(float))
                                           : static_cast<NSUInteger>(t) *
                                                 static_cast<NSUInteger>(static_cast<size_t>(batch) *
                                                                         static_cast<size_t>(input_size) * sizeof(float));
      const NSUInteger y_off = batch_first ? static_cast<NSUInteger>(t) * static_cast<NSUInteger>(hidden_size * sizeof(float))
                                           : static_cast<NSUInteger>(t) *
                                                 static_cast<NSUInteger>(static_cast<size_t>(batch) *
                                                                         static_cast<size_t>(hidden_size) * sizeof(float));

      MPSMatrixDescriptor* x_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(batch)
                                                                         columns:static_cast<NSUInteger>(input_size)
                                                                        rowBytes:(batch_first ? static_cast<NSUInteger>(seq_len * input_size * sizeof(float))
                                                                                              : static_cast<NSUInteger>(input_size * sizeof(float)))
                                                                        dataType:MPSDataTypeFloat32];
      MPSMatrixDescriptor* y_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(batch)
                                                                         columns:static_cast<NSUInteger>(hidden_size)
                                                                        rowBytes:(batch_first ? static_cast<NSUInteger>(seq_len * hidden_size * sizeof(float))
                                                                                              : static_cast<NSUInteger>(hidden_size * sizeof(float)))
                                                                        dataType:MPSDataTypeFloat32];

      MPSMatrix* x_mat = [[MPSMatrix alloc] initWithBuffer:x_buf offset:x_off descriptor:x_desc];
      MPSMatrix* y_mat = [[MPSMatrix alloc] initWithBuffer:out_buf offset:y_off descriptor:y_desc];
      [src_mats addObject:x_mat];
      [dst_mats addObject:y_mat];
    }

    NSMutableArray<MPSRNNRecurrentMatrixState*>* states = [NSMutableArray array];
    [layer encodeSequenceToCommandBuffer:cb
                          sourceMatrices:src_mats
                     destinationMatrices:dst_mats
                     recurrentInputState:nil
                   recurrentOutputStates:states];

    // Extract final state matrices and copy into the output buffers.
    if (states.count > 0) {
      MPSRNNRecurrentMatrixState* last = [states objectAtIndex:0];
      MPSMatrix* h_mat = [last getRecurrentOutputMatrixForLayerIndex:0];
      MPSMatrix* c_mat = [last getMemoryCellMatrixForLayerIndex:0];

      id<MTLBuffer> h_dst = (__bridge id<MTLBuffer>)(h_n->data);
      id<MTLBuffer> c_dst = (__bridge id<MTLBuffer>)(c_n->data);

      const NSUInteger state_bytes = static_cast<NSUInteger>(out_state_elems * sizeof(float));
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit copyFromBuffer:h_mat.data sourceOffset:0 toBuffer:h_dst destinationOffset:0 size:state_bytes];
      [blit copyFromBuffer:c_mat.data sourceOffset:0 toBuffer:c_dst destinationOffset:0 size:state_bytes];
      [blit endEncoding];
    }

    [cb commit];
  });
}

}  // namespace contrib
}  // namespace tvm
