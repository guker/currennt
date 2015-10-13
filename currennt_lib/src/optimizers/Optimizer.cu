/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include "Optimizer.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../layers/BinaryClassificationLayer.hpp"
#include "../layers/MulticlassClassificationLayer.hpp"
#include "../Configuration.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/getRawPointer.cuh"

#include <limits>

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace thrust {
    template <typename floatT>
    struct elasticnet_dfunctor {
        const floatT a;
        const floatT b;
        elasticnet_dfunctor(floatT _a, floatT _b) : a(_a), b(_b) {}

        __host__ __device__
            floatT operator()(const floatT& x, const floatT& y) const {
                return a * x + copysign(b, x) + y;
            }
    };

    template <typename floatT>
    struct elasticnet_functor {
        const floatT a;
        const floatT b;
        elasticnet_functor(floatT _a, floatT _b) : a(_a), b(_b) {}

        __host__ __device__
            floatT operator()(const floatT& x) const {
                return a * x * x + b * fabs(x);
            }
    };

}

namespace optimizers {
    template <typename TDevice>
    real_t Optimizer<TDevice>::_processDataSet(data_sets::DataSet &ds, bool calcWeightUpdates, real_t *classError)
    {
        // process all data set fractions
        real_t error = 0;
        *classError = (real_t) ds.totalTimesteps();

        boost::shared_ptr<data_sets::DataSetFraction> frac;
        bool firstFraction = true;
        while ((frac = ds.getNextFraction())) {
            // compute forward pass and calculate the error
            m_neuralNetwork.loadSequences(*frac);
            m_neuralNetwork.computeForwardPass();
            error += m_neuralNetwork.calculateError();

            if (dynamic_cast<layers::BinaryClassificationLayer<TDevice>*>(&m_neuralNetwork.postOutputLayer()))
                *classError -= (real_t)static_cast<layers::BinaryClassificationLayer<TDevice>&>(m_neuralNetwork.postOutputLayer()).countCorrectClassifications();
            if (dynamic_cast<layers::MulticlassClassificationLayer<TDevice>*>(&m_neuralNetwork.postOutputLayer()))
                *classError -= (real_t)static_cast<layers::MulticlassClassificationLayer<TDevice>&>(m_neuralNetwork.postOutputLayer()).countCorrectClassifications();
            
            if (calcWeightUpdates) {
                // weight noise:
                std::vector<Cpu::real_vector> origWeights(m_neuralNetwork.layers().size());
                if (Configuration::instance().weightNoiseSigma() > 0.0) {
                    for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
                        layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
                        if (layer) {
                            origWeights[i] = layer->weights();
                            layer->injectWeightNoise(Configuration::instance().weightNoiseSigma());
                        }
                    }
                }
                // compute the backward pass and accumulate the weight updates
                m_neuralNetwork.computeBackwardPass();

                for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
                    layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
                    if (!layer)
                        continue;

                    if (!firstFraction && !Configuration::instance().hybridOnlineBatch())
                        thrust::transform(layer->weightUpdates().begin(), layer->weightUpdates().end(), m_curWeightUpdates[i].begin(), m_curWeightUpdates[i].begin(), thrust::plus<real_t>());
                    else
                    	thrust::copy(layer->weightUpdates().begin(), layer->weightUpdates().end(), m_curWeightUpdates[i].begin());

                    // restore old weights before update in case of weight noise
                    if (Configuration::instance().weightNoiseSigma() > 0.0)
                        thrust::copy(origWeights[i].begin(), origWeights[i].end(), layer->weights().begin());
                }

                // update weights for hybrid online/batch learning
                if (Configuration::instance().hybridOnlineBatch()){
                    _updateWeights(error);
                }
            }

            firstFraction = false;
        }

        // Add elastic-net penalty terms
        for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
            layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (!layer)
                continue;
            thrust::transform(layer->weights().begin(), layer->weights().end(), m_curWeightUpdates[i].begin(), m_curWeightUpdates[i].begin(), thrust::elasticnet_dfunctor<real_t>(m_alpha, m_beta));
            error += thrust::transform_reduce(layer->weights().begin(), layer->weights().end(), thrust::elasticnet_functor<real_t>(m_alpha, m_beta), 0.0, thrust::plus<real_t>());
        }

        // normalize the errors
        error /= ds.totalSequences();
        *classError /= (real_t)ds.totalTimesteps();

        // update weights for batch learning
        if (calcWeightUpdates && !Configuration::instance().hybridOnlineBatch()){
            _updateWeights(error);
        }

        return error;
    }

    template <typename TDevice>
    void Optimizer<TDevice>::_exportWeights(const helpers::JsonDocument &jsonDoc, const char *arrayName, const std::vector<real_vector> &weights)
    {
        rapidjson::Value weightsArray(rapidjson::kArrayType);
        weightsArray.Reserve((rapidjson::SizeType)weights.size(), jsonDoc->GetAllocator());

        for (size_t i = 0; i < weights.size(); ++i) {
            rapidjson::Value v(rapidjson::kArrayType);
            Cpu::real_vector w = weights[i];
            v.Reserve((rapidjson::SizeType)w.size(), jsonDoc->GetAllocator());
            for (size_t j = 0; j < w.size(); ++j)
                v.PushBack(w[j], jsonDoc->GetAllocator());
            weightsArray.PushBack(v, jsonDoc->GetAllocator());
        }

        jsonDoc->AddMember(arrayName, weightsArray, jsonDoc->GetAllocator());
    }

    template <typename TDevice>
    void Optimizer<TDevice>::_importWeights(const helpers::JsonDocument &jsonDoc, const char *arrayName, std::vector<real_vector> *weights)
    {
        if (!jsonDoc->HasMember(arrayName) || !(*jsonDoc)[arrayName].IsArray())
            throw std::runtime_error(std::string("Array '") + arrayName + "' is missing or has the wrong type");

        if ((*jsonDoc)[arrayName].Size() != (rapidjson::SizeType)weights->size())
            throw std::runtime_error(std::string("Array '") + arrayName + "' has a wrong size");

        int i = 0;
        for (rapidjson::Value::ConstValueIterator it = (*jsonDoc)[arrayName].Begin(); it != (*jsonDoc)[arrayName].End(); ++it) {
            if (!it->IsArray())
                throw std::runtime_error(std::string("Object in '") + arrayName + "' is not an array");
            if (it->Size() != (rapidjson::SizeType)(*weights)[i].size())
                throw std::runtime_error(std::string("Subarray in '") + arrayName + "' has a wrong size");

            Cpu::real_vector w;
            w.reserve(it->Size());
            for (rapidjson::Value::ConstValueIterator it2 = it->Begin(); it2 != it->End(); ++it2)
                w.push_back((real_t)it2->GetDouble());

            (*weights)[i] = w;

            ++i;
        }
    }

    template <typename TDevice>
    void Optimizer<TDevice>::storeWeights(std::vector<real_vector>& x)
    {
        for (size_t i = 1; i < m_neuralNetwork.layers().size() - 1; ++i) {
            layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (layer){
            	thrust::copy(layer->weights().begin(), layer->weights().end(), x[i].begin());
            }
        }
    }

    template <typename TDevice>
    void Optimizer<TDevice>::_restoreWeights(const std::vector<real_vector>& x)
    {
        for (size_t i = 1; i < m_neuralNetwork.layers().size() - 1; ++i) {
            layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (layer){
            	thrust::copy(x[i].begin(), x[i].end(), layer->weights().begin());
            }
        }
    }

    template <typename TDevice>
    void Optimizer<TDevice>::resetWeights(void)
    {
        _restoreWeights(m_lastWeights);
    }


    template <typename TDevice>
    NeuralNetwork<TDevice>& Optimizer<TDevice>::_neuralNetwork()
    {
        return m_neuralNetwork;
    }

    template <typename TDevice>
    const std::vector<typename Optimizer<TDevice>::real_vector>& Optimizer<TDevice>::_curWeightUpdates() const
    {
        return m_curWeightUpdates;
    }

    template <typename TDevice>
    Optimizer<TDevice>::Optimizer(NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, 
                                   data_sets::DataSet &validationSet, data_sets::DataSet &testSet,
                                   int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery,
                                   real_t alpha, real_t beta)
        : m_neuralNetwork             (neuralNetwork)
        , m_trainingSet               (trainingSet)
        , m_validationSet             (validationSet)
        , m_testSet                   (testSet)
        , m_maxEpochs                 (maxEpochs)
        , m_maxEpochsNoBest           (maxEpochsNoBest)
        , m_validateEvery             (validateEvery)
        , m_testEvery                 (testEvery)
        , m_finished                  (false)
        , m_curEpoch                  (0)
        , m_epochsSinceLowestError    (0)
        , m_lowestValidationError     (std::numeric_limits<real_t>::max())
        , m_curTrainingError          (std::numeric_limits<real_t>::max())
        , m_curValidationError        (std::numeric_limits<real_t>::max())
        , m_curTestError              (std::numeric_limits<real_t>::max())
        , m_curValidationClassError   (0)
        , m_curTrainingClassError     (0)
        , m_curTestClassError         (0)
        , m_alpha                     (alpha)
        , m_beta                      (beta)
    {
        // initialize the last and best weights vectors
        m_lastWeights.resize(m_neuralNetwork.layers().size());
        m_bestWeights.resize(m_neuralNetwork.layers().size());
        for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
       	    layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (layer){
                m_lastWeights[i] = layer->weights();
                m_bestWeights[i] = layer->weights();
            }
        }

        // initialize the current weight updates vectors
        m_curWeightUpdates = m_bestWeights;
    }

    template <typename TDevice>
    Optimizer<TDevice>::~Optimizer()
    {
    }

    template <typename TDevice>
    bool Optimizer<TDevice>::finished() const
    {
        return m_finished;
    }

    template <typename TDevice>
    int Optimizer<TDevice>::currentEpoch() const
    {
        return m_curEpoch;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::lowestValidationError() const
    {
        return m_lowestValidationError;
    }

    template <typename TDevice>
    int Optimizer<TDevice>::epochsSinceLowestValidationError() const
    {
        return m_epochsSinceLowestError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curTrainingError() const
    {
        return m_curTrainingError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curValidationError() const
    {
        return m_curValidationError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curTestError() const
    {
        return m_curTestError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curTrainingClassError() const
    {
        return m_curTrainingClassError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curValidationClassError() const
    {
        return m_curValidationClassError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curTestClassError() const
    {
        return m_curTestClassError;
    }

    template <typename TDevice>
    bool Optimizer<TDevice>::train()
    {
        if (!m_finished) {
            ++m_curEpoch;

            // train one epoch and update the weights
            m_curTrainingError = _processDataSet(m_trainingSet, true, &m_curTrainingClassError);

            // calculate the validation error and store the weights if we a new lowest error
            if (!m_validationSet.empty() && m_curEpoch % m_validateEvery == 0) {
                m_curValidationError = _processDataSet(m_validationSet, false, &m_curValidationClassError);
                
                if (m_curValidationError < m_lowestValidationError) {
                    m_lowestValidationError  = m_curValidationError;
                    m_epochsSinceLowestError = 0;

                    storeWeights(m_bestWeights);
                }
                else {
                    m_epochsSinceLowestError += m_validateEvery;
                }
            }
            else if (m_validationSet.empty()) {
                m_epochsSinceLowestError = 0;
                storeWeights(m_bestWeights);
            }

            // calculate the test error
            if (!m_testSet.empty() && m_curEpoch % m_testEvery == 0)
                m_curTestError = _processDataSet(m_testSet, false, &m_curTestClassError);

            // check if we did not get a new lowest error for some training epochs 
            // or if we reached the maximum number of training epochs
            if (m_epochsSinceLowestError >= m_maxEpochsNoBest || (m_maxEpochs >= 0 && m_curEpoch >= m_maxEpochs)) {
                _restoreWeights(m_bestWeights);
                m_finished = true;
            }
        }

        return m_finished;
    }

    template <typename TDevice>
    void Optimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        jsonDoc->AddMember("optimizer_finished",                   m_finished,                jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_epoch",                  m_curEpoch,                jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_epochs_since_lowest_error",  m_epochsSinceLowestError,  jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_lowest_validation_error",    m_lowestValidationError,   jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_training_error",         m_curTrainingError,        jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_validation_error",       m_curValidationError,      jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_test_error",             m_curTestError,            jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_training_class_error",   m_curTrainingClassError,   jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_validation_class_error", m_curValidationClassError, jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_test_class_error",       m_curTestClassError,       jsonDoc->GetAllocator());

        _exportWeights(jsonDoc, "optimizer_best_weights", m_bestWeights);
    }

    template <typename TDevice>
    void Optimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        m_finished                = helpers::checkedJsonGet<bool  >(*jsonDoc, "optimizer_finished");
        m_curEpoch                = helpers::checkedJsonGet<int   >(*jsonDoc, "optimizer_cur_epoch");
        m_epochsSinceLowestError  = helpers::checkedJsonGet<int   >(*jsonDoc, "optimizer_epochs_since_lowest_error");
        m_lowestValidationError   = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_lowest_validation_error");
        m_curTrainingError        = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_training_error");
        m_curValidationError      = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_validation_error");
        m_curTestError            = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_test_error");
        m_curTrainingClassError   = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_training_class_error");
        m_curValidationClassError = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_validation_class_error");
        m_curTestClassError       = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_test_class_error");

        _importWeights(jsonDoc, "optimizer_best_weights", &m_bestWeights);
    }


    // explicit template instantiations
    template class Optimizer<Cpu>;
    template class Optimizer<Gpu>;

} // namespace optimizers
