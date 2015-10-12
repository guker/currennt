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

#include "../../currennt_lib/src/Configuration.hpp"
#include "../../currennt_lib/src/NeuralNetwork.hpp"
#include "../../currennt_lib/src/layers/LstmLayer.hpp"
#include "../../currennt_lib/src/layers/BinaryClassificationLayer.hpp"
#include "../../currennt_lib/src/layers/MulticlassClassificationLayer.hpp"
#include "../../currennt_lib/src/optimizers/SteepestDescentOptimizer.hpp"
#include "../../currennt_lib/src/helpers/JsonClasses.hpp"
#include "../../currennt_lib/src/rapidjson/prettywriter.h"
#include "../../currennt_lib/src/rapidjson/filestream.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <stdarg.h>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <csignal>


void swap32 (uint32_t *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 3 ); *( q + 3 ) = temp;
  temp = *( q + 1 ); *( q + 1 ) = *( q + 2 ); *( q + 2 ) = temp;
}

void swap16 (uint16_t *p) 
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 1 ); *( q + 1 ) = temp;
}

void swapFloat(float *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 3 ); *( q + 3 ) = temp;
  temp = *( q + 1 ); *( q + 1 ) = *( q + 2 ); *( q + 2 ) = temp;
}

enum data_set_type
{
    DATA_SET_TRAINING,
    DATA_SET_VALIDATION,
    DATA_SET_TEST,
    DATA_SET_FEEDFORWARD
};

// helper functions (implementation below)
void readJsonFile(rapidjson::Document *doc, const std::string &filename);
boost::shared_ptr<data_sets::DataSet> loadDataSet(data_set_type dsType);
template <typename TDevice> void fprintLayers(FILE * fp, const NeuralNetwork<TDevice> &nn);
template <typename TDevice> void fprintOptimizer(FILE * fp, const optimizers::Optimizer<TDevice> &optimizer);
template <typename TDevice> void saveNetwork(const NeuralNetwork<TDevice> &nn, const std::string &filename);
void createModifiedTrainingSet(data_sets::DataSet *trainingSet, int parallelSequences, bool outputsToClasses, boost::mutex &swapTrainingSetsMutex);
template <typename TDevice> void saveState(const NeuralNetwork<TDevice> &nn, const optimizers::Optimizer<TDevice> &optimizer, const std::string &infoRows);
template <typename TDevice> void restoreState(NeuralNetwork<TDevice> *nn, optimizers::Optimizer<TDevice> *optimizer, std::string *infoRows);
std::string fprintfRow(FILE * fp, const char *format, ...);

void catch_signal(int signalNumber){
	throw(std::runtime_error("Caught SIGINT"));
}


// main function
template <typename TDevice>
int trainerMain(const Configuration &config)
{
    boost::shared_ptr<data_sets::DataSet> trainingSet    = boost::make_shared<data_sets::DataSet>();
    boost::shared_ptr<data_sets::DataSet> validationSet  = boost::make_shared<data_sets::DataSet>();
    boost::shared_ptr<data_sets::DataSet> testSet        = boost::make_shared<data_sets::DataSet>();
    boost::shared_ptr<data_sets::DataSet> feedForwardSet = boost::make_shared<data_sets::DataSet>();

    try {
        signal(SIGINT, catch_signal);
        // read the neural network description file 
        std::string networkFile = config.continueFile().empty() ? config.networkFile() : config.continueFile();
        fprintf( stderr, "Reading network from '%s'... ", networkFile.c_str());
        fflush(stderr);
        rapidjson::Document netDoc;
        readJsonFile(&netDoc, networkFile);
        fprintf( stderr, "done.\n");
        fprintf( stderr, "\n");

        // load data sets
        if (config.trainingMode()) {
            trainingSet = loadDataSet(DATA_SET_TRAINING);
            
            if (!config.validationFiles().empty())
                validationSet = loadDataSet(DATA_SET_VALIDATION);
            
            if (!config.testFiles().empty())
                testSet = loadDataSet(DATA_SET_TEST);
        }
        else {
            feedForwardSet = loadDataSet(DATA_SET_FEEDFORWARD);
        }

        // calculate the maximum sequence length
        int maxSeqLength;
        if (config.trainingMode())
            maxSeqLength = std::max(trainingSet->maxSeqLength(), std::max(validationSet->maxSeqLength(), testSet->maxSeqLength()));
        else
            maxSeqLength = feedForwardSet->maxSeqLength();

        int parallelSequences = config.parallelSequences();
       
        // modify input and output size in netDoc to match the training set size 
        // trainingSet->inputPatternSize
        // trainingSet->outputPatternSize

        // create the neural network
        fprintf(stderr, "Creating the neural network... ");
        fflush(stdout);
        int inputSize = -1;
        int outputSize = -1;
        inputSize = trainingSet->inputPatternSize();
        outputSize = trainingSet->outputPatternSize();
        NeuralNetwork<TDevice> neuralNetwork(netDoc, parallelSequences, maxSeqLength, inputSize, outputSize);

        if (!trainingSet->empty() && trainingSet->outputPatternSize() != neuralNetwork.postOutputLayer().size())
            throw std::runtime_error("Post output layer size != target pattern size of the training set");
        if (!validationSet->empty() && validationSet->outputPatternSize() != neuralNetwork.postOutputLayer().size())
            throw std::runtime_error("Post output layer size != target pattern size of the validation set");
        if (!testSet->empty() && testSet->outputPatternSize() != neuralNetwork.postOutputLayer().size())
            throw std::runtime_error("Post output layer size != target pattern size of the test set");

        fprintf(stderr, "done.\n");
        fprintf(stderr, "Layers:\n");
        fprintLayers(stderr, neuralNetwork);
        fprintf(stderr, "\n");

        // check if this is a classification task
        bool classificationTask = false;
        if (dynamic_cast<layers::BinaryClassificationLayer<TDevice>*>(&neuralNetwork.postOutputLayer())
            || dynamic_cast<layers::MulticlassClassificationLayer<TDevice>*>(&neuralNetwork.postOutputLayer())) {
                classificationTask = true;
        }

        fprintf(stderr, "\n");

        // create the optimizer
        if (config.trainingMode()) {
            fprintf(stderr, "Creating the optimizer... ");
            fflush(stdout);
            boost::scoped_ptr<optimizers::Optimizer<TDevice> > optimizer;
            optimizers::SteepestDescentOptimizer<TDevice> *sdo;

            switch (config.optimizer()) {
            case Configuration::OPTIMIZER_STEEPESTDESCENT:
                sdo = new optimizers::SteepestDescentOptimizer<TDevice>(
                    neuralNetwork, *trainingSet, *validationSet, *testSet,
                    config.maxEpochs(), config.maxEpochsNoBest(), config.validateEvery(), config.testEvery(),
                    config.learningRate(), config.momentum(), config.alpha(), config.beta()
                    );
                optimizer.reset(sdo);
                break;

            default:
                throw std::runtime_error("Unknown optimizer type");
            }

            fprintf(stderr, "done.\n");
            fprintOptimizer(stderr, config, *optimizer);

            std::string infoRows;

            // continue from autosave?
            if (!config.continueFile().empty()) {
                fprintf(stderr, "Restoring state from '%s'... ", config.continueFile().c_str());
                fflush(stdout);
                restoreState(&neuralNetwork, &*optimizer, &infoRows);
                fprintf(stderr, "done.\n\n");
            }

            // train the network
            fprintf(stderr, "Starting training...\n");
            fprintf(stderr, "\n");

            if (classificationTask){
                printf("Epoch\tDuration\tTrainMisclassified\tTrainingError\tValidationMisclassified\tValidationError\tTestMisclassified\tTestError\tNewBest\n");
            } else {
                printf("Epoch\tDuration\tTrainingError\tValidationError\tTestError\tNewBest\n");
            }
            std::cout << infoRows;

            bool finished = false;
            while (!finished) {
                const char *errFormat = (classificationTask ? "%.2lf\t%.2f\t" : "%f\t");
                const char *errSpace  = (classificationTask ? "\t\t" : "\t");

                // train for one epoch and measure the time
                infoRows += fprintfRow(stdout, "%d\t", optimizer->currentEpoch() + 1);
                
                boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::local_time();
                finished = optimizer->train();
                boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
                double duration = (double)(endTime - startTime).total_milliseconds() / 1000.0;

                infoRows += fprintfRow(stdout, "%.1lf\t", duration);
                if (classificationTask)
                    infoRows += fprintfRow(stdout, errFormat, (double)optimizer->curTrainingClassError()*100.0, (double)optimizer->curTrainingError());
                else
                    infoRows += fprintfRow(stdout, errFormat, (double)optimizer->curTrainingError());
                
                if (!validationSet->empty() && optimizer->currentEpoch() % config.validateEvery() == 0) {
                    if (classificationTask)
                        infoRows += fprintfRow(stdout, errFormat, (double)optimizer->curValidationClassError()*100.0, (double)optimizer->curValidationError());
                    else
                        infoRows += fprintfRow(stdout, errFormat, (double)optimizer->curValidationError());
                }
                else
                    infoRows += fprintfRow(stdout, "%s", errSpace);

                if (!testSet->empty() && optimizer->currentEpoch() % config.testEvery() == 0) {
                    if (classificationTask)
                        infoRows += fprintfRow(stdout, errFormat, (double)optimizer->curTestClassError()*100.0, (double)optimizer->curTestError());
                    else
                        infoRows += fprintfRow(stdout, errFormat, (double)optimizer->curTestError());
                }
                else
                    infoRows += fprintfRow(stdout, "%s", errSpace);

                if (!validationSet->empty() && optimizer->currentEpoch() % config.validateEvery() == 0) {
                    if (optimizer->epochsSinceLowestValidationError() == 0) {
                        infoRows += fprintfRow(stdout, "yes\n");
                        if (config.autosaveBest()) {
                            std::stringstream saveFileS;
                            if (config.autosavePrefix().empty()) {
                                size_t pos = config.networkFile().find_last_of('.');
                                if (pos != std::string::npos && pos > 0)
                                    saveFileS << config.networkFile().substr(0, pos);
                                else
                                    saveFileS << config.networkFile();
                            }
                            else
                                saveFileS << config.autosavePrefix();
                            saveFileS << ".best.jsn";
                            saveNetwork(neuralNetwork, saveFileS.str());
                        }
                    }
                    else
                        infoRows += fprintfRow(stdout, "no\n");
                }
                else
                    infoRows += fprintfRow(stdout, "\n");

                // autosave
                if (config.autosave())
                    saveState(neuralNetwork, *optimizer, infoRows);
            }

            fprintf(stderr, "\n");

            if (optimizer->epochsSinceLowestValidationError() == config.maxEpochsNoBest())
                fprintf(stderr, "No new lowest error since %d epochs. Training stopped.\n", config.maxEpochsNoBest());
            else
                fprintf(stderr, "Maximum number of training epochs reached. Training stopped.\n");

            if (!validationSet->empty())
                fprintf(stderr, "Lowest validation error: %lf\n", optimizer->lowestValidationError());
            else
                fprintf(stderr, "Final training set error: %lf\n", optimizer->curTrainingError());
            fprintf(stderr, "\n");

            // save the trained network to the output file
            fprintf(stderr, "Storing the trained network in '%s'... ", config.trainedNetworkFile().c_str());
            saveNetwork(neuralNetwork, config.trainedNetworkFile());
            fprintf(stderr, "done.\n");

            std::cerr << "Removing cache file(s) ..." << std::endl;
            if (trainingSet != boost::shared_ptr<data_sets::DataSet>())
                boost::filesystem::remove(trainingSet->cacheFileName());
            if (validationSet != boost::shared_ptr<data_sets::DataSet>())
                boost::filesystem::remove(validationSet->cacheFileName());
            if (testSet != boost::shared_ptr<data_sets::DataSet>())
                boost::filesystem::remove(testSet->cacheFileName());
        }
        // evaluation mode
        else {
            Cpu::real_vector outputMeans  = feedForwardSet->outputMeans();
            Cpu::real_vector outputStdevs = feedForwardSet->outputStdevs();
            assert (outputMeans.size()  == feedForwardSet->outputPatternSize());
            assert (outputStdevs.size() == feedForwardSet->outputPatternSize());
            //for (int i = 0; i < outputMeans.size(); ++i) 
             //   printf("outputMeans[%d] = %f outputStdevs[%d] = %f\n", i, outputMeans[i], i, outputStdevs[i]);
            bool unstandardize = config.revertStd(); 
            if (unstandardize) {
                fprintf(stderr, "Outputs will be scaled by mean and standard deviation specified in NC file.\n");
            }

            int output_lag = config.outputTimeLag();

            if (config.feedForwardFormat() == Configuration::FORMAT_SINGLE_CSV) {
                // open the output file
                std::ofstream file(config.feedForwardOutputFile().c_str(), std::ofstream::out);

                // process all data set fractions
                int fracIdx = 0;
                boost::shared_ptr<data_sets::DataSetFraction> frac;
                while (((frac = feedForwardSet->getNextFraction()))) {
                    fprintf(stderr, "Computing outputs for data fraction %d...", ++fracIdx);
                    fflush(stderr);

                    // compute the forward pass for the current data fraction and extract the outputs
                    neuralNetwork.loadSequences(*frac);
                    neuralNetwork.computeForwardPass();
                    std::vector<std::vector<std::vector<real_t> > > outputs = neuralNetwork.getOutputs();

                    // write the outputs in the file
                    for (int psIdx = 0; psIdx < (int)outputs.size(); ++psIdx) {
                        // write the sequence tag
                        file << frac->seqInfo(psIdx).seqTag;

                        // write the patterns
                        for (int timestep = 0; timestep < (int)outputs[psIdx].size(); ++timestep) {
                            for (int outputIdx = 0; outputIdx < (int)outputs[psIdx][timestep].size(); ++outputIdx) {
                                real_t v;
                                if (timestep < outputs[psIdx].size() - output_lag)
                                    v = outputs[psIdx][timestep + output_lag][outputIdx];
                                else
                                    v = outputs[psIdx][outputs[psIdx].size() - 1][outputIdx];
                                if (unstandardize) {
                                    v *= outputStdevs[outputIdx];
                                    v += outputMeans[outputIdx];
                                }
                                file << ';' << v; 
                            }
                        }

                        file << '\n';
                    }

                    fprintf(stderr, " done.\n");
                }

                // close the file
                file.close();
            } // format: FORMAT_SINGLE_CSV

            else if (config.feedForwardFormat() == Configuration::FORMAT_CSV) {
                // process all data set fractions
                int fracIdx = 0;
                boost::shared_ptr<data_sets::DataSetFraction> frac;
                while (((frac = feedForwardSet->getNextFraction()))) {
                    fprintf(stderr, "Computing outputs for data fraction %d...", ++fracIdx);
                    fflush(stderr);

                    // compute the forward pass for the current data fraction and extract the outputs
                    neuralNetwork.loadSequences(*frac);
                    neuralNetwork.computeForwardPass();
                    std::vector<std::vector<std::vector<real_t> > > outputs = neuralNetwork.getOutputs();

                    // write one output file per sequence
                    for (int psIdx = 0; psIdx < (int)outputs.size(); ++psIdx) {
                        boost::filesystem::path seqPath(frac->seqInfo(psIdx).seqTag);
                        seqPath.replace_extension(".csv");
                        std::string filename(seqPath.filename().string());
                        boost::filesystem::path oPath = boost::filesystem::path(config.feedForwardOutputFile()) / seqPath.relative_path().parent_path();
                        boost::filesystem::create_directories(oPath);
                        boost::filesystem::path filepath = oPath / filename;
                        std::ofstream file(filepath.string().c_str(), std::ofstream::out);

                        // write the patterns
                        for (int timestep = 0; timestep < (int)outputs[psIdx].size(); ++timestep) {
                            for (int outputIdx = 0; outputIdx < (int)outputs[psIdx][timestep].size(); ++outputIdx) {
                                real_t v;
                                if (timestep < outputs[psIdx].size() - output_lag)
                                    v = outputs[psIdx][timestep + output_lag][outputIdx];
                                else
                                    v = outputs[psIdx][outputs[psIdx].size() - 1][outputIdx];
                                if (unstandardize) {
                                    v *= outputStdevs[outputIdx];
                                    v += outputMeans[outputIdx];
                                }
                                if (outputIdx > 0)
                                    file << ';';
                                file << v; 
                            }
                            file << '\n';
                        }
                        file.close();
                    }

                    fprintf(stderr, " done.\n");
                }
            } // format: FORMAT_CSV

            else if (config.feedForwardFormat() == Configuration::FORMAT_HTK) {
                // process all data set fractions
                int fracIdx = 0;
                boost::shared_ptr<data_sets::DataSetFraction> frac;
                while (((frac = feedForwardSet->getNextFraction()))) {
                    fprintf(stderr, "Computing outputs for data fraction %d...", ++fracIdx);
                    fflush(stderr);

                    // compute the forward pass for the current data fraction and extract the outputs
                    neuralNetwork.loadSequences(*frac);
                    neuralNetwork.computeForwardPass();
                    std::vector<std::vector<std::vector<real_t> > > outputs = neuralNetwork.getOutputs();

                    // write one output file per sequence
                    for (int psIdx = 0; psIdx < (int)outputs.size(); ++psIdx) {
                        if (outputs[psIdx].size() > 0) {
                            // replace_extension does not work in all Boost versions ...
                            //std::string seqTag = frac->seqInfo(psIdx).seqTag;
                            /*size_t dot_pos = seqTag.find_last_of('.');
                            if (dot_pos != std::string::npos && dot_pos > 0) {
                                seqTag = seqTag.substr(0, dot_pos);
                            }*/
                            //seqTag += ".htk";
                            //std::cout << seqTag << std::endl;
                            boost::filesystem::path seqPath(frac->seqInfo(psIdx).seqTag + ".htk");
                            std::string filename(seqPath.filename().string());
                            boost::filesystem::path oPath = boost::filesystem::path(config.feedForwardOutputFile()) / seqPath.relative_path().parent_path();
                            boost::filesystem::create_directories(oPath);
                            boost::filesystem::path filepath = oPath / filename;
                            std::ofstream file(filepath.string().c_str(), std::ofstream::out | std::ios::binary);

                            int nComps = outputs[psIdx][0].size();

                            // write header
                            unsigned tmp = (unsigned)outputs[psIdx].size();
                            swap32(&tmp);
                            file.write((const char*)&tmp, sizeof(unsigned));
                            tmp = (unsigned)(config.featurePeriod() * 1e4);
                            swap32(&tmp);
                            file.write((const char*)&tmp, sizeof(unsigned));
                            unsigned short tmp2 = (unsigned short)(nComps) * sizeof(float);
                            swap16(&tmp2);
                            file.write((const char*)&tmp2, sizeof(unsigned short));
                            tmp2 = (unsigned short)(config.outputFeatureKind());
                            swap16(&tmp2);
                            file.write((const char*)&tmp2, sizeof(unsigned short));

                            float v;
                            // write the patterns
                            for (int timestep = 0; timestep < (int)outputs[psIdx].size(); ++timestep) {
                                for (int outputIdx = 0; outputIdx < (int)outputs[psIdx][timestep].size(); ++outputIdx) {
                                    float v;
                                    if (timestep < outputs[psIdx].size() - output_lag)
                                        v = (float)outputs[psIdx][timestep + output_lag][outputIdx];
                                    else
                                        v = (float)outputs[psIdx][outputs[psIdx].size() - 1][outputIdx];
                                    if (unstandardize) {
                                        v *= outputStdevs[outputIdx];
                                        v += outputMeans[outputIdx];
                                    }
                                    swapFloat(&v); 
                                    file.write((const char*)&v, sizeof(float));
                                }
                            }
                            file.close();
                        }
                    }

                    fprintf(stderr, " done.\n");
                }
            }
            if (feedForwardSet != boost::shared_ptr<data_sets::DataSet>()) 
                std::cerr << "Removing cache file: " << feedForwardSet->cacheFileName() << std::endl;
            boost::filesystem::remove(feedForwardSet->cacheFileName());
        } // evaluation mode
    }
    catch (const std::exception &e) {
        fprintf(stderr, "FAILED: %s\n", e.what());
        std::cerr << "Removing cache file(s) ..." << std::endl;
        if (trainingSet != boost::shared_ptr<data_sets::DataSet>())
            boost::filesystem::remove(trainingSet->cacheFileName());
        if (validationSet != boost::shared_ptr<data_sets::DataSet>())
            boost::filesystem::remove(validationSet->cacheFileName());
        if (testSet != boost::shared_ptr<data_sets::DataSet>())
            boost::filesystem::remove(testSet->cacheFileName());
        if (feedForwardSet != boost::shared_ptr<data_sets::DataSet>())
            boost::filesystem::remove(feedForwardSet->cacheFileName());
        return 2;
    }

    return 0;
}


int main(int argc, const char *argv[])
{
    // load the configuration
    Configuration config(argc, argv);

    // Crude signal handling
    //signal(SIGINT, catch_signal);

    // run the execution device specific main function
    if (config.useCuda()) {
        int count;
        cudaError_t err;
        if (config.listDevices()) {
            if ((err = cudaGetDeviceCount(&count)) != cudaSuccess) {
                std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
                return err;
            }
            std::cerr << count << " devices found" << std::endl;
            cudaDeviceProp prop;
            for (int i = 0; i < count; ++i) {
                if ((err = cudaGetDeviceProperties(&prop, i)) != cudaSuccess) {
                    std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
                    return err;
                }
                std::cerr << i << ": " << prop.name << std::endl;
            }
            return 0;
        }
        int device = 0;
        char* dev = std::getenv("CURRENNT_CUDA_DEVICE");
        if (dev != NULL) {
            device = std::atoi(dev);
        }
        cudaDeviceProp prop;
        if ((err = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) {
            std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        std::cerr << "Using device #" << device << " (" << prop.name << ")" << std::endl;
        if ((err = cudaSetDevice(device)) != cudaSuccess) {
            std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        return trainerMain<Gpu>(config);
    }
    else
        return trainerMain<Cpu>(config);
}


void readJsonFile(rapidjson::Document *doc, const std::string &filename)
{
    // open the file
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    if (!ifs.good())
        throw std::runtime_error("Cannot open file");
 
    // calculate the file size in bytes
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    // read the file into a buffer
    char *buffer = new char[size + 1];
    ifs.read(buffer, size);
    buffer[size] = '\0';

    std::string docStr(buffer);
    delete buffer;

    // extract the JSON tree
    if (doc->Parse<0>(docStr.c_str()).HasParseError())
        throw std::runtime_error(std::string("Parse error: ") + doc->GetParseError());
}


boost::shared_ptr<data_sets::DataSet> loadDataSet(data_set_type dsType)
{
    std::string type;
    std::vector<std::string> filenames;
    real_t fraction = 1;
    bool fracShuf   = false;
    bool seqShuf    = false;
    real_t noiseDev = 0;
    std::string cachePath = "";
    int truncSeqLength = -1;

    cachePath = Configuration::instance().cachePath();
    switch (dsType) {
    case DATA_SET_TRAINING:
        type     = "training set";
        filenames = Configuration::instance().trainingFiles();
        fraction = Configuration::instance().trainingFraction();
        fracShuf = Configuration::instance().shuffleFractions();
        seqShuf  = Configuration::instance().shuffleSequences();
        noiseDev = Configuration::instance().inputNoiseSigma();
        truncSeqLength = Configuration::instance().truncateSeqLength();
        break;

    case DATA_SET_VALIDATION:
        type     = "validation set";
        filenames = Configuration::instance().validationFiles();
        fraction = Configuration::instance().validationFraction();
        cachePath = Configuration::instance().cachePath();
        truncSeqLength = Configuration::instance().truncateValSeqLength();
        break;

    case DATA_SET_TEST:
        type     = "test set";
        filenames = Configuration::instance().testFiles();
        fraction = Configuration::instance().testFraction();
        break;

    default:
        type     = "feed forward input set";
        filenames = Configuration::instance().feedForwardInputFiles();
        noiseDev = Configuration::instance().inputNoiseSigma();
        break;
    }

    fprintf(stderr, "Loading %s ", type.c_str());
    for (std::vector<std::string>::const_iterator fn_itr = filenames.begin();
         fn_itr != filenames.end(); ++fn_itr)
    {
        fprintf(stderr, "'%s' ", fn_itr->c_str());
    }
    fprintf(stderr, "...");
    fflush(stderr);

    //std::cout << "truncating to " << truncSeqLength << std::endl;
    boost::shared_ptr<data_sets::DataSet> ds = boost::make_shared<data_sets::DataSet>(
        filenames,
        Configuration::instance().parallelSequences(), fraction, truncSeqLength, 
        fracShuf, seqShuf, noiseDev, cachePath);

    fprintf(stderr, "done.\n");
    fprintf(stderr, "Loaded fraction:  %d%%\n",   (int)(fraction*100));
    fprintf(stderr, "Sequences:        %d\n",     ds->totalSequences());
    fprintf(stderr, "Sequence lengths: %d..%d\n", ds->minSeqLength(), ds->maxSeqLength());
    fprintf(stderr, "Total timesteps:  %d\n",     ds->totalTimesteps());
    fprintf(stderr, "\n");

    return ds;
}


template <typename TDevice>
void fprintLayers(FILE * fp, const NeuralNetwork<TDevice> &nn)
{
    int weights = 0;

    for (int i = 0; i < (int)nn.layers().size(); ++i) {
        fprintf(fp, "(%d) %s ", i, nn.layers()[i]->type().c_str());
        fprintf(fp, "[size: %d", nn.layers()[i]->size());

        const layers::TrainableLayer<TDevice>* tl = dynamic_cast<const layers::TrainableLayer<TDevice>*>(nn.layers()[i].get());
        if (tl) {
            fprintf(fp, ", bias: %.1lf, weights: %d", (double)tl->bias(), (int)tl->weights().size());
            weights += (int)tl->weights().size();
        }

        fprintf(fp, "]\n");
    }

    fprintf(fp, "Total weights: %d\n", weights);
}


template <typename TDevice> 
void fprintOptimizer(FILE * fp, const Configuration &config, const optimizers::Optimizer<TDevice> &optimizer)
{
    if (dynamic_cast<const optimizers::SteepestDescentOptimizer<TDevice>*>(&optimizer)) {
        fprintf(fp, "Optimizer type: Steepest descent with momentum\n");
        fprintf(fp, "Max training epochs:       %d\n", config.maxEpochs());
        fprintf(fp, "Max epochs until new best: %d\n", config.maxEpochsNoBest());
        fprintf(fp, "Validation error every:    %d\n", config.validateEvery());
        fprintf(fp, "Test error every:          %d\n", config.testEvery());
        fprintf(fp, "Learning rate:             %g\n", (double)config.learningRate());
        fprintf(fp, "Momentum:                  %g\n", (double)config.momentum());
        fprintf(fp, "Elastic net:               %g %g\n", (double)config.alpha(), (double)config.beta());
        fprintf(fp, "\n");
    }
}


template <typename TDevice> 
void saveNetwork(const NeuralNetwork<TDevice> &nn, const std::string &filename)
{
    rapidjson::Document jsonDoc;
    jsonDoc.SetObject();
    nn.exportLayers (&jsonDoc);
    nn.exportWeights(&jsonDoc);

    FILE *file = fopen(filename.c_str(), "w");
    if (!file)
        throw std::runtime_error("Cannot open file");

    rapidjson::FileStream os(file);
    rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
    jsonDoc.Accept(writer);

    fclose(file);
}


template <typename TDevice> 
void saveState(const NeuralNetwork<TDevice> &nn, const optimizers::Optimizer<TDevice> &optimizer, const std::string &infoRows)
{
    // create the JSON document
    rapidjson::Document jsonDoc;
    jsonDoc.SetObject();

    // add the configuration options
    jsonDoc.AddMember("configuration", Configuration::instance().serializedOptions().c_str(), jsonDoc.GetAllocator());

    // add the info rows
    std::string tmp = boost::replace_all_copy(infoRows, "\n", ";;;");
    jsonDoc.AddMember("info_rows", tmp.c_str(), jsonDoc.GetAllocator());

    // add the network structure and weights
    nn.exportLayers (&jsonDoc);
    nn.exportWeights(&jsonDoc);

    // add the state of the optimizer
    optimizer.exportState(&jsonDoc);
    
    // open the file
    std::stringstream autosaveFilename;
    std::string prefix = Configuration::instance().autosavePrefix(); 
    autosaveFilename << prefix;
    if (!prefix.empty())
        autosaveFilename << '_';
    autosaveFilename << "epoch";
    autosaveFilename << std::setfill('0') << std::setw(3) << optimizer.currentEpoch();
    autosaveFilename << ".autosave";
    std::string autosaveFilename_str = autosaveFilename.str();
    FILE *file = fopen(autosaveFilename_str.c_str(), "w");
    if (!file)
        throw std::runtime_error("Cannot open file");

    // write the file
    rapidjson::FileStream os(file);
    rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
    jsonDoc.Accept(writer);
    fclose(file);
}


template <typename TDevice> 
void restoreState(NeuralNetwork<TDevice> *nn, optimizers::Optimizer<TDevice> *optimizer, std::string *infoRows)
{
    rapidjson::Document jsonDoc;
    readJsonFile(&jsonDoc, Configuration::instance().continueFile());

    // extract info rows
    if (!jsonDoc.HasMember("info_rows"))
        throw std::runtime_error("Missing value 'info_rows'");
    *infoRows = jsonDoc["info_rows"].GetString();
    boost::replace_all(*infoRows, ";;;", "\n");

    // extract the state of the optimizer
    optimizer->importState(jsonDoc);
}


std::string fprintfRow(FILE * fp, const char *format, ...)
{
    // write to temporary buffer
    char buffer[100];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    fputs(buffer, fp);
    fflush(fp);

    // return the same string
    return std::string(buffer);
}
