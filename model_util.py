#!/usr/bin/env python
#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

from __future__ import print_function
def get_layer_weights(weights_file=None, layer_name=None):
    '''
    Copy from https://github.com/Arsey/keras-transfer-learning-for-oxford102/blob/master/models/vgg16.py
    '''
    if not weights_file or not layer_name:
        return None
    else:
        ## print type(weights_file)
        ## print weights_file.keys()
        if len(weights_file.keys())==0:
            print('Empty weights file: ', weights_file)
        g = weights_file[layer_name]
        weights = [g[p] for p in g]
        if layer_name.find('batchnormalization')>=0:
            # gamma beta order is reversed...
            weights = [weights[1], weights[0], weights[2], weights[3]]
        
        ## print 'Weights for "{}" are loaded'.format(layer_name)
        return weights