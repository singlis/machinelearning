// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.LightGBM;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Conversions;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        /// <summary>
        /// FastTreeBinaryClassification TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeBinaryEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            var trainer = ML.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryClassificationTrainer.Options
                {
                    NumThreads = 1,
                    NumTrees = 10,
                    NumLeaves = 5,
                });

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGBMBinaryEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            var trainer = ML.BinaryClassification.Trainers.LightGbm(new Options
            {
                NumLeaves = 10,
                NThread = 1,
                MinDataPerLeaf = 2,
            });

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }


        class MonotoneTestData
        {
            [VectorType(11)]
            public float[] Features { get; set; }
        }

        class MonotoneTestResult
        {
            public float Score = 0.0F;
        }


        private MonotoneTestData[] GenerateTestFeatures(int index, int featureCount)
        {
            float x = 0.0F;
            MonotoneTestData[] testData = new MonotoneTestData[100];
            for (int i = 0; i < 100; ++i)
            {
                testData[i] = new MonotoneTestData()
                {
                    Features = new float[featureCount]
                };
                testData[i].Features[index] = x;
                x += 0.1F;
            }

            return testData;
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGBMEstimatorTestMonotonic()
        {
            // Create a sample dataset to verify that the constraint is working
            var mlContext = new MLContext();
            var dataView = GetRegressionPipeline();
            var trainer = mlContext.Regression.Trainers.LightGbm(new Options() { MonotoneConstraints = new string[] { "pos:0", "neg:1" } }).Fit(dataView);
            var engine = trainer.CreatePredictionEngine<MonotoneTestData, MonotoneTestResult>(Env);

            // Feature 0 is monotonically increasing and feature 1 is monotonically decreasing. 
            // This confirms the these trends are true by generating test data for the positive feature
            // and a test data for the negatively feature, then calls predict with each test data 
            // confirming the increasing and decreasing.
            int numFeatures = 11;
            var positiveTestData = GenerateTestFeatures(0, numFeatures);
            var negativeTestData = GenerateTestFeatures(1, numFeatures);

            var lastValue = float.MinValue;
            foreach (var x in positiveTestData)
            {
                var result = engine.Predict(x);
                Assert.True(result.Score >= lastValue);
                lastValue = result.Score;
            }

            lastValue = float.MaxValue;
            foreach (var x in negativeTestData)
            {
                var result = engine.Predict(x);
                Assert.True(result.Score <= lastValue);
                lastValue = result.Score;
            }
        }

        [Fact]
        public void LightGBMMonotoneArgumentWithRangeAndSingle()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos:0-10,15,17,18-20" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 21);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,1,1", constraintString);
        }

        [Fact]
        public void LightGBMMonotoneArgumentWithRangeOverlapping()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos:0-10", "neg:5-8" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("1,1,1,1,1,-1,-1,-1,-1,1,1,0,0,0,0,0,0,0,0,0", constraintString);
        }

        [Fact]
        public void LightGBMMonotoneArgumentWithPositiveAndNegative()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos:0-5", "neg:7-8,15" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("1,1,1,1,1,1,0,-1,-1,0,0,0,0,0,0,-1,0,0,0,0", constraintString);
        }

        [Fact]
        public void LightGBMMonotoneArgumentInvalid()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos1:2" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0", constraintString);
        }

        [Fact]
        public void LightGBMMonotoneArgumentInvalidFormat()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos1:2:3" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0", constraintString);
        }

        [Fact]
        public void LightGBMMonotoneArgumentPartialInvalid()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos1:2", "neg:4" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0", constraintString);
        }


        [Fact]
        public void LightGBMMonotoneArgumentInvalidIndex()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos1:200" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0", constraintString);
        }

        [Fact]
        public void LightGBMMonotoneArgumentNegativeIndex()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos1:-200" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0", constraintString);
        }

        [Fact]
        public void LightGBMMonotoneArgumentInvalidRange()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos1:10-1" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0", constraintString);
        }


        [Fact]
        public void LightGBMMonotoneArgumentNull()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = null;
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Null(constraintString);
        }

        [Fact]
        public void LightGBMMonotoneArgumentBadValue()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos:TabbyCat" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0", constraintString);
        }

        [Fact]
        public void LightGBMMonotoneArgumentAllPositive()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "pos" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1", constraintString);
        }

        [Fact]
        public void LightGBMMonotoneArgumentAllNegative()
        {
            Dictionary<string, object> options = new Dictionary<string, object>();
            options["monotone_constraints"] = new string[] { "neg" };
            var trainer = new LightGbmBinaryTrainer(Env, new Options() { });
            trainer.ExpandMonotoneConstraint(ref options, 20);
            Assert.True(options.ContainsKey("monotone_constraints"));
            var constraintString = options["monotone_constraints"];
            Assert.Equal("-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1", constraintString);
        }


        [Fact]
        public void GAMClassificationEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            var trainer = new BinaryClassificationGamTrainer(Env, new BinaryClassificationGamTrainer.Options
            {
                GainConfidenceLevel = 0,
                NumIterations = 15,
            });
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }


        [Fact]
        public void FastForestClassificationEstimator()
        {
            var (pipe, dataView) = GetBinaryClassificationPipeline();

            var trainer = ML.BinaryClassification.Trainers.FastForest(
                new FastForestClassification.Options
                {
                    NumLeaves = 10,
                    NumTrees = 20,
                });

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }

        /// <summary>
        /// FastTreeRankingTrainer TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeRankerEstimator()
        {
            var (pipe, dataView) = GetRankingPipeline();

            var trainer = ML.Ranking.Trainers.FastTree(
                new FastTreeRankingTrainer.Options
                {
                    FeatureColumn = "NumericFeatures",
                    NumTrees = 10
                });

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }

        /// <summary>
        /// LightGbmRankingTrainer TrainerEstimator test 
        /// </summary>
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGBMRankerEstimator()
        {
            var (pipe, dataView) = GetRankingPipeline();

            var trainer = ML.Ranking.Trainers.LightGbm(labelColumn: "Label0", featureColumn: "NumericFeatures", groupIdColumn: "Group", learningRate: 0.4);

            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Train(transformedDataView, transformedDataView);
            Done();
        }

        /// <summary>
        /// FastTreeRegressor TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.FastTree(
                new FastTreeRegressionTrainer.Options { NumTrees = 10, NumThreads = 1, NumLeaves = 5 });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Train(dataView, dataView);
            Done();
        }

        /// <summary>
        /// LightGbmRegressorTrainer TrainerEstimator test 
        /// </summary>
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGBMRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.LightGbm(new Options
            {
                NThread = 1,
                NormalizeFeatures = NormalizeOption.Warn,
                CatL2 = 5,
            });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Train(dataView, dataView);
            Done();
        }


        /// <summary>
        /// RegressionGamTrainer TrainerEstimator test 
        /// </summary>
        [Fact]
        public void GAMRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = new RegressionGamTrainer(Env, new RegressionGamTrainer.Options
            {
                EnablePruning = false,
                NumIterations = 15,
            });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Train(dataView, dataView);
            Done();
        }

        /// <summary>
        /// FastTreeTweedieTrainer TrainerEstimator test 
        /// </summary>
        [Fact]
        public void TweedieRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.FastTreeTweedie(
                new FastTreeTweedieTrainer.Options
                {
                    EntropyCoefficient = 0.3,
                    OptimizationAlgorithm = BoostedTreeArgs.OptimizationAlgorithmType.AcceleratedGradientDescent,
                });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Train(dataView, dataView);
            Done();
        }

        /// <summary>
        /// FastForestRegression TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastForestRegressorEstimator()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.FastForest(
                new FastForestRegression.Options
                {
                    BaggingSize = 2,
                    NumTrees = 10,
                });

            TestEstimatorCore(trainer, dataView);
            var model = trainer.Train(dataView, dataView);
            Done();
        }

        /// <summary>
        /// LightGbmMulticlass TrainerEstimator test 
        /// </summary>
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGbmMultiClassEstimator()
        {
            var (pipeline, dataView) = GetMultiClassPipeline();
            var trainer = ML.MulticlassClassification.Trainers.LightGbm(learningRate: 0.4);
            var pipe = pipeline.Append(trainer)
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        // Number of examples
        private const int _rowNumber = 1000;
        // Number of features
        private const int _columnNumber = 5;
        // Number of classes
        private const int _classNumber = 3;
        private class GbmExample
        {
            [VectorType(_columnNumber)]
            public float[] Features;
            [KeyType(Count = _classNumber)]
            public uint Label;
            [VectorType(_classNumber)]
            public float[] Score;
        }

        private void LightGbmHelper(bool useSoftmax, out string modelString, out List<GbmExample> mlnetPredictions, out double[] lgbmRawScores, out double[] lgbmProbabilities)
        {
            // Prepare data and train LightGBM model via ML.NET
            // Training matrix. It contains all feature vectors.
            var dataMatrix = new float[_rowNumber * _columnNumber];
            // Labels for multi-class classification
            var labels = new uint[_rowNumber];
            // Training list, which is equivalent to the training matrix above.
            var dataList = new List<GbmExample>();
            for (/*row index*/ int i = 0; i < _rowNumber; ++i)
            {
                int featureSum = 0;
                var featureVector = new float[_columnNumber];
                for (/*column index*/ int j = 0; j < _columnNumber; ++j)
                {
                    int featureValue = (j + i * _columnNumber) % 10;
                    featureSum += featureValue;
                    dataMatrix[j + i * _columnNumber] = featureValue;
                    featureVector[j] = featureValue;
                }
                labels[i] = (uint)featureSum % _classNumber;
                dataList.Add(new GbmExample { Features = featureVector, Label = labels[i], Score = new float[_classNumber] });
            }

            var mlContext = new MLContext(seed: 0, conc: 1);
            var dataView = mlContext.Data.ReadFromEnumerable(dataList);
            int numberOfTrainingIterations = 3;
            var gbmTrainer = new LightGbmMulticlassTrainer(mlContext, new Options
            {
                NumBoostRound = numberOfTrainingIterations,
                MinDataPerGroup = 1,
                MinDataPerLeaf = 1,
                UseSoftmax = useSoftmax
            });

            var gbm = gbmTrainer.Fit(dataView);
            var predicted = gbm.Transform(dataView);
            mlnetPredictions = mlContext.CreateEnumerable<GbmExample>(predicted, false).ToList();

            // Convert training to LightGBM's native format and train LightGBM model via its APIs
            // Convert the whole training matrix to CSC format required by LightGBM interface. Notice that the training matrix
            // is dense so this conversion is simply a matrix transpose.
            double[][] sampleValueGroupedByColumn = new double[_columnNumber][];
            int[][] sampleIndicesGroupedByColumn = new int[_columnNumber][];
            int[] sampleNonZeroCntPerColumn = new int[_columnNumber];
            for (int j = 0; j < _columnNumber; ++j)
            {
                // Allocate memory for the j-th column in the training matrix
                sampleValueGroupedByColumn[j] = new double[_rowNumber];
                sampleIndicesGroupedByColumn[j] = new int[_rowNumber];
                sampleNonZeroCntPerColumn[j] = _rowNumber;
                // Copy the j-th column in training matrix
                for (int i = 0; i < _rowNumber; ++i)
                {
                    // data[j + i * _columnNumber] is the value at the j-th column and the i-th row.
                    sampleValueGroupedByColumn[j][i] = dataMatrix[j + i * _columnNumber];
                    // Row index of the assigned value.
                    sampleIndicesGroupedByColumn[j][i] = i;
                }
            }

            // LightGBM only accepts float labels.
            float[] floatLabels = new float[_rowNumber];
            for (int i = 0; i < _rowNumber; ++i)
                floatLabels[i] = labels[i];

            // Allocate LightGBM data container (called Dataset in LightGBM world).
            var gbmDataSet = new Dataset(sampleValueGroupedByColumn, sampleIndicesGroupedByColumn, _columnNumber, sampleNonZeroCntPerColumn, _rowNumber, _rowNumber, "", floatLabels);

            // Push training examples into LightGBM data container.
            gbmDataSet.PushRows(dataMatrix, _rowNumber, _columnNumber, 0);

            // Probability output.
            lgbmProbabilities = new double[_rowNumber * _classNumber];
            // Raw score.
            lgbmRawScores = new double[_rowNumber * _classNumber];

            // Get parameters used in ML.NET's LightGBM
            var gbmParams = gbmTrainer.GetGbmParameters();

            // Call LightGBM C-style APIs to do prediction.
            modelString = null;
            using (var ch = (mlContext as IChannelProvider).Start("Training LightGBM..."))
            using (var pch = (mlContext as IProgressChannelProvider).StartProgressChannel("Training LightGBM..."))
            {
                var host = (mlContext as IHostEnvironment).Register("Training LightGBM...");
                var gbmNative = WrappedLightGbmTraining.Train(ch, pch, gbmParams, gbmDataSet, numIteration: numberOfTrainingIterations);

                int nativeLength = 0;
                unsafe
                {
                    fixed (float* data = dataMatrix)
                    fixed (double* result0 = lgbmProbabilities)
                    fixed (double* result1 = lgbmRawScores)
                    {
                        WrappedLightGbmInterface.BoosterPredictForMat(gbmNative.Handle, (IntPtr)data, WrappedLightGbmInterface.CApiDType.Float32,
                            _rowNumber, _columnNumber, 1, (int)WrappedLightGbmInterface.CApiPredictType.Normal, numberOfTrainingIterations, "", ref nativeLength, result0);
                        WrappedLightGbmInterface.BoosterPredictForMat(gbmNative.Handle, (IntPtr)data, WrappedLightGbmInterface.CApiDType.Float32,
                            _rowNumber, _columnNumber, 1, (int)WrappedLightGbmInterface.CApiPredictType.Raw, numberOfTrainingIterations, "", ref nativeLength, result1);
                    }
                    modelString = gbmNative.GetModelString();
                }
            }
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGbmMultiClassEstimatorCompareOva()
        {
            // Train ML.NET LightGBM and native LightGBM and apply the trained models to the training set.
            LightGbmHelper(useSoftmax: false, out string modelString, out List<GbmExample> mlnetPredictions, out double[] nativeResult1, out double[] nativeResult0);

            // The i-th predictor returned by LightGBM produces the raw score, denoted by z_i, of the i-th class.
            // Assume that we have n classes in total. The i-th class probability can be computed via
            // p_i = sigmoid(sigmoidScale * z_i) / (sigmoid(sigmoidScale * z_1) + ... + sigmoid(sigmoidScale * z_n)).
            Assert.True(modelString != null);
            float sigmoidScale = 0.5f; // Constant used train LightGBM. See gbmParams["sigmoid"] in the helper function.
            // Compare native LightGBM's and ML.NET's LightGBM results example by example
            for (int i = 0; i < _rowNumber; ++i)
            {
                double sum = 0;
                for (int j = 0; j < _classNumber; ++j)
                {
                    Assert.Equal(nativeResult0[j + i * _classNumber], mlnetPredictions[i].Score[j], 6);
                    sum += MathUtils.SigmoidSlow(sigmoidScale * (float)nativeResult1[j + i * _classNumber]);
                }
                for (int j = 0; j < _classNumber; ++j)
                {
                    double prob = MathUtils.SigmoidSlow(sigmoidScale * (float)nativeResult1[j + i * _classNumber]);
                    Assert.Equal(prob / sum, mlnetPredictions[i].Score[j], 6);
                }
            }

            Done();
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGbmMultiClassEstimatorCompareSoftMax()
        {
            // Train ML.NET LightGBM and native LightGBM and apply the trained models to the training set.
            LightGbmHelper(useSoftmax: true, out string modelString, out List<GbmExample> mlnetPredictions, out double[] nativeResult1, out double[] nativeResult0);

            // The i-th predictor returned by LightGBM produces the raw score, denoted by z_i, of the i-th class.
            // Assume that we have n classes in total. The i-th class probability can be computed via
            // p_i = exp(z_i) / (exp(z_1) + ... + exp(z_n)).
            Assert.True(modelString != null);
            // Compare native LightGBM's and ML.NET's LightGBM results example by example
            for (int i = 0; i < _rowNumber; ++i)
            {
                double sum = 0;
                for (int j = 0; j < _classNumber; ++j)
                {
                    Assert.Equal(nativeResult0[j + i * _classNumber], mlnetPredictions[i].Score[j], 6);
                    sum += Math.Exp((float)nativeResult1[j + i * _classNumber]);
                }
                for (int j = 0; j < _classNumber; ++j)
                {
                    double prob = Math.Exp(nativeResult1[j + i * _classNumber]);
                    Assert.Equal(prob / sum, mlnetPredictions[i].Score[j], 6);
                }
            }

            Done();
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void LightGbmInDifferentCulture()
        {
            var currentCulture = Thread.CurrentThread.CurrentCulture;
            Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("de-DE");
            var (pipeline, dataView) = GetMultiClassPipeline();
            var trainer = ML.MulticlassClassification.Trainers.LightGbm(learningRate: 0.4);
            var pipe = pipeline.Append(trainer)
                    .Append(ML.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var model = pipe.Fit(dataView);
            var metrics = ML.MulticlassClassification.Evaluate(model.Transform(dataView));
            Assert.True(metrics.AccuracyMacro > 0.8);
            Thread.CurrentThread.CurrentCulture = currentCulture;
        }
    }
}
