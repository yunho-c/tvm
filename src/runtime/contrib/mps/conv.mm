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
#include "mps_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tvm.contrib.mps.buffer2img",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    auto buf = args[0].cast<DLTensor*>();
                    auto img = args[1].cast<DLTensor*>();
                    // copy to temp
                    id<MTLBuffer> mtlbuf = (__bridge id<MTLBuffer>)(buf->data);
                    MetalThreadEntry* entry_ptr = MetalThreadEntry::ThreadLocal();
                    runtime::metal::MetalThreadEntry* rt =
                        runtime::metal::MetalThreadEntry::ThreadLocal();
                    id<MTLDevice> dev = entry_ptr->metal_api->GetDevice(buf->device);
                    id<MTLBuffer> temp = rt->GetTempBuffer(buf->device, [mtlbuf length]);
                    runtime::metal::Stream* stream = entry_ptr->metal_api->CastStreamOrGetDefault(
                        entry_ptr->metal_api->GetCurrentStream(buf->device), buf->device.device_id);
                    id<MTLCommandBuffer> cb =
                        stream->GetCommandBuffer("tvm.contrib.mps.buffer2img.copy");
                    id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
                    [encoder copyFromBuffer:mtlbuf
                               sourceOffset:0
                                   toBuffer:temp
                          destinationOffset:0
                                       size:[mtlbuf length]];
                    [encoder endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];

                    MPSImageDescriptor* desc = [MPSImageDescriptor
                        imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                   width:buf->shape[2]
                                                  height:buf->shape[1]
                                         featureChannels:buf->shape[3]];

                    MPSImage* mpsimg = entry_ptr->AllocMPSImage(dev, desc);

                    [mpsimg writeBytes:[temp contents]
                            dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
                            imageIndex:0];

                    img->data = (__bridge void*)mpsimg;

                    [mpsimg readBytes:[temp contents]
                           dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
                           imageIndex:0];
                  })
      .def_packed("tvm.contrib.mps.img2buffer",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    auto img = args[0].cast<DLTensor*>();
                    auto buf = args[1].cast<DLTensor*>();
                    id<MTLBuffer> mtlbuf = (__bridge id<MTLBuffer>)(buf->data);
                    MPSImage* mpsimg = (__bridge MPSImage*)(img->data);
                    MetalThreadEntry* entry_ptr = MetalThreadEntry::ThreadLocal();
                    runtime::metal::MetalThreadEntry* rt =
                        runtime::metal::MetalThreadEntry::ThreadLocal();
                    id<MTLBuffer> temp = rt->GetTempBuffer(buf->device, [mtlbuf length]);

                    [mpsimg readBytes:[temp contents]
                           dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
                           imageIndex:0];

                    runtime::metal::Stream* stream = entry_ptr->metal_api->CastStreamOrGetDefault(
                        entry_ptr->metal_api->GetCurrentStream(buf->device), buf->device.device_id);
                    id<MTLCommandBuffer> cb =
                        stream->GetCommandBuffer("tvm.contrib.mps.img2buffer.copy");
                    id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
                    [encoder copyFromBuffer:temp
                               sourceOffset:0
                                   toBuffer:mtlbuf
                          destinationOffset:0
                                       size:[mtlbuf length]];
                    [encoder endEncoding];
                    [cb commit];
                  })
      .def_packed("tvm.contrib.mps.conv2d", [](ffi::PackedArgs args, ffi::Any* ret) {
        // MPS-NHWC
        auto data = args[0].cast<DLTensor*>();
        auto weight = args[1].cast<DLTensor*>();
        auto output = args[2].cast<DLTensor*>();
        int pad = args[3].cast<int>();
        int stride = args[4].cast<int>();

        ICHECK_EQ(data->ndim, 4);
        ICHECK_EQ(weight->ndim, 4);
        ICHECK_EQ(output->ndim, 4);
        ICHECK(ffi::IsContiguous(*output));
        ICHECK(ffi::IsContiguous(*weight));
        ICHECK(ffi::IsContiguous(*data));

        ICHECK_EQ(data->shape[0], 1);
        ICHECK_EQ(output->shape[0], 1);

        int oCh = weight->shape[0];
        int kH = weight->shape[1];
        int kW = weight->shape[2];
        int iCh = weight->shape[3];

        const auto f_buf2img = tvm::ffi::Function::GetGlobal("tvm.contrib.mps.buffer2img");
        const auto f_img2buf = tvm::ffi::Function::GetGlobal("tvm.contrib.mps.img2buffer");
        // Get Metal device API
        MetalThreadEntry* entry_ptr = MetalThreadEntry::ThreadLocal();
        runtime::metal::MetalThreadEntry* rt = runtime::metal::MetalThreadEntry::ThreadLocal();
        id<MTLDevice> dev = entry_ptr->metal_api->GetDevice(data->device);
        runtime::metal::Stream* stream = entry_ptr->metal_api->CastStreamOrGetDefault(
            entry_ptr->metal_api->GetCurrentStream(data->device), data->device.device_id);
        id<MTLCommandBuffer> cb = stream->GetCommandBuffer("tvm.contrib.mps.conv2d");
        // data to MPSImage
        DLTensor tmp_in;
        (*f_buf2img)(data, &tmp_in);
        MPSImage* tempA = (__bridge MPSImage*)tmp_in.data;
        // weight to temp memory
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)(weight->data);
        id<MTLBuffer> tempB = rt->GetTempBuffer(weight->device, [bufB length]);
        {
          id<MTLCommandBuffer> copy_cb =
              stream->GetCommandBuffer("tvm.contrib.mps.conv2d.copy_weight");
          id<MTLBlitCommandEncoder> encoder = [copy_cb blitCommandEncoder];
          [encoder copyFromBuffer:bufB
                     sourceOffset:0
                         toBuffer:tempB
                destinationOffset:0
                             size:[bufB length]];
          [encoder endEncoding];
          [copy_cb commit];
          [copy_cb waitUntilCompleted];
        }
        float* ptr_w = (float*)[tempB contents];
        // output to MPSImage
        DLTensor tmp_out;
        (*f_buf2img)(output, &tmp_out);
        MPSImage* tempC = (__bridge MPSImage*)tmp_out.data;
        // conv desc

        MPSCNNConvolutionDescriptor* conv_desc =
            [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kW
                                                                    kernelHeight:kH
                                                            inputFeatureChannels:iCh
                                                           outputFeatureChannels:oCh];
        [conv_desc setStrideInPixelsX:stride];
        [conv_desc setStrideInPixelsY:stride];

        MPSCNNConvolution* conv =
            [[MPSCNNConvolution alloc] initWithDevice:dev
                                convolutionDescriptor:conv_desc
                                        kernelWeights:ptr_w
                                            biasTerms:nil
                                                flags:MPSCNNConvolutionFlagsNone];
        if (pad == 0) {
          conv.padding = [MPSNNDefaultPadding
              paddingWithMethod:MPSNNPaddingMethodAddRemainderToTopLeft |
                                MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodSizeSame];
        } else if (pad == 1) {
          conv.padding = [MPSNNDefaultPadding
              paddingWithMethod:MPSNNPaddingMethodAddRemainderToTopLeft |
                                MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodSizeValidOnly];
        }
        [conv encodeToCommandBuffer:cb sourceImage:tempA destinationImage:tempC];

        id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
        [encoder synchronizeResource:tempC.texture];
        [encoder endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        (*f_img2buf)(&tmp_out, output);
      });
}

}  // namespace contrib
}  // namespace tvm
