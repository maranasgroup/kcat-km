import tensorflow as tf

try:
    device = tf.config.PhysicalDevice('GPU:0', device_type='GPU')
    tf.config.experimental.set_memory_growth(device, True)
except:
    pass

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import numpy as np
import json
from tqdm import tqdm

class Conv1DPredictor(object):
    
    def ConvLayer(self, kernel_size, filters, activation, initializer, regularizer_param, padding='same'):
        def f(input):
            model_p = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
            model_p = BatchNormalization()(model_p)
            model_p = Activation(activation)(model_p)
            return GlobalMaxPooling1D()(model_p)
        return f

    def build_model(self, 
            sequence_feat_name = 'integer',
            smiles_feat_name = 'fp_morgan3',
            fc_dropout=0, substrate_fc_layers=256,
            protein_kernel_sizes="[5,10,15]", protein_filters=32, 
            output_fc_layers=256, protein_feat_type=None, 
            protein_feat_length=1024, protein_feat_depth=25, 
            activation='relu', protein_fc_layers=None, 
            regularizer_param = 0.001, initializer="glorot_normal", 
            substrate_feat_length=2048, substrate_fc_dropout=0.0,
            protein_embedding_dropout = 0.2, protein_fc_dropout = 0.0, 
            output_fc_dropout = 0.0, integer_embedding_depth = 20, 
            learning_rate=0.0001, batch_size=64, epochs = 10, 
            loss='mean_squared_error'):
                
        def _return_tuple(value):
            if type(value) is int:
                return [value]
            else:
                return tuple(value)
        
        protein_kernel_sizes = json.loads(protein_kernel_sizes)

        input_sub = Input(shape=(substrate_feat_length,)) # Yeah, this works for fixed length substrates only 
        
        params_dic = {"kernel_initializer": initializer,
                      "kernel_regularizer": l2(regularizer_param)}
        
        input_layer_sub = input_sub
        if substrate_fc_layers is not None:
            substrate_fc_layers = _return_tuple(substrate_fc_layers)
            for layer_size in substrate_fc_layers:
                model_sub = Dense(layer_size, **params_dic)(input_layer_sub)
                model_sub = BatchNormalization()(model_sub)
                model_sub = Activation(activation)(model_sub)
                model_sub = Dropout(substrate_fc_dropout)(model_sub)
                input_layer_sub = model_sub

        if protein_feat_type == "Integer_Embedding":
            # print('Integer EMBEDDING'*100)
            input_prot = Input(shape=(protein_feat_length)) # For integer embedding, no depth, depth is chosen in the embedding layer below
            model_prot = Embedding(26,integer_embedding_depth, embeddings_initializer=initializer,embeddings_regularizer=l2(regularizer_param))(input_prot)
            model_prot = SpatialDropout1D(protein_embedding_dropout)(model_prot)
            
        elif protein_feat_type == "Learned_Embedding":
            input_prot = Input(shape=(protein_feat_length,protein_feat_depth)) # For learned embedding, both length and depth are specified
            model_prot = input_prot
            
        model_convs = [self.ConvLayer(kernel_size, protein_filters, activation, initializer, regularizer_param)(model_prot) for kernel_size in protein_kernel_sizes]
        
        # if there are more than one conv layers, concatenate them side by side
        if len(model_convs)!=1:
            model_prot = Concatenate(axis=1)(model_convs)
        else:
            model_prot = model_convs[0]
        
        # if there are fc layers for protein
        if protein_fc_layers:
            input_layer_prot = model_prot
            protein_fc_layers = _return_tuple(protein_fc_layers)
            for layer in protein_fc_layers:
                model_prot = Dense(layer, **params_dic)(input_layer_prot)
                model_prot = BatchNormalization()(model_prot)
                model_prot = Activation(activation)(model_prot)
                model_prot = Dropout(protein_fc_dropout)(model_prot)
                input_layer_prot = model_prot

        model_output = Concatenate(axis=1)([model_sub,model_prot])
        if output_fc_layers is not None:
            output_fc_layers = _return_tuple(output_fc_layers)
            for fc_layer in output_fc_layers:
                model_output = Dense(units=fc_layer, **params_dic)(model_output)
                model_output = BatchNormalization()(model_output)
                model_output = Activation(activation)(model_output)
                model_output = Dropout(output_fc_dropout)(model_output)
                # input_dim = fc_layer
                
        model_output = Dense(1, activation='linear', activity_regularizer=l2(regularizer_param),**params_dic)(model_output)
        model = Model(inputs=[input_sub, input_prot], outputs = model_output)

        return model
    
    def __init__(self, params):
        self.params = params
        self.modelobj = self.build_model(**params)
        opt = Adam(learning_rate=params["learning_rate"])
        loss = tf.keras.losses.MeanAbsoluteError(reduction="auto", name="mean_squared_error")
        self.modelobj.compile(optimizer=opt, loss=loss, metrics=['mean_squared_error', 'mean_absolute_error'],run_eagerly=True)
        
    def summary(self):
        self.modelobj.summary()

    def predict(self, substrate_input, sequence_input):
        return self.modelobj.predict(substrate_input, sequence_input)

    def save_model(self, model_path):
        self.modelobj.save(model_path, weights_only=True)

    def load_model(self, model_path):
        self.modelobj.load_weights(model_path)