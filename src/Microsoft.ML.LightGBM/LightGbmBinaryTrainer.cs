// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.LightGBM;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(LightGbmBinaryTrainer.Summary, typeof(LightGbmBinaryTrainer), typeof(Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    LightGbmBinaryTrainer.UserName, LightGbmBinaryTrainer.LoadNameValue, LightGbmBinaryTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(LightGbmBinaryModelParameters), null, typeof(SignatureLoadModel),
    "LightGBM Binary Executor",
    LightGbmBinaryModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(LightGbm), null, typeof(SignatureEntryPointModule), "LightGBM")]

namespace Microsoft.ML.LightGBM
{
    public sealed class LightGbmBinaryModelParameters : TreeEnsembleModelParametersBasedOnRegressionTree
    {
        internal const string LoaderSignature = "LightGBMBinaryExec";
        internal const string RegistrationName = "LightGBMBinaryPredictor";

        private static VersionInfo GetVersionInfo()
        {
            // REVIEW: can we decouple the version from FastTree predictor version ?
            return new VersionInfo(
                modelSignature: "LGBBINCL",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                //verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, // Categorical splits.
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LightGbmBinaryModelParameters).Assembly.FullName);
        }

        private protected override uint VerNumFeaturesSerialized => 0x00010002;
        private protected override uint VerDefaultValueSerialized => 0x00010004;
        private protected override uint VerCategoricalSplitSerialized => 0x00010005;
        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        internal LightGbmBinaryModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private LightGbmBinaryModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var predictor = new LightGbmBinaryModelParameters(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new ValueMapperCalibratedModelParameters<LightGbmBinaryModelParameters, ICalibrator>(env, predictor, calibrator);
        }
    }

    ///<summary>
    /// Trains a Light GBM Model.
    ///</summary>
    /// <remarks>
    /// Light GBM is an open source implementation of boosted trees.
    /// <a href = 'https://github.com/Microsoft/LightGBM/wiki' > GitHub: LightGBM</a>
    /// </remarks>
    public sealed class LightGbmBinaryTrainer : LightGbmTrainerBase<LightGbmBinaryTrainer.Options,float,
        BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>,
        CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>
    {
        internal const string UserName = "LightGBM Binary Classifier";
        internal const string LoadNameValue = "LightGBMBinary";
        internal const string ShortName = "LightGBM";
        internal const string Summary = "Train a LightGBM binary classification model.";

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public sealed class Options : OptionsBase
        {
            public enum EvaluateMetricType
            {
                Default,
                Logloss,
                Error,
                Auc,
            };

            [Argument(ArgumentType.AtMostOnce, HelpText = "Parameter for the sigmoid function." + nameof(LightGbmBinaryTrainer) + ", " + nameof(LightGbmMulticlassTrainer) +
                " and in " + nameof(LightGbmRankingTrainer) + ".", ShortName = "sigmoid")]
            [TGUI(Label = "Sigmoid", SuggestedSweeps = "0.5,1")]
            public double Sigmoid = 0.5;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Evaluation metrics.",
                ShortName = "em")]
            public EvaluateMetricType EvaluationMetric = EvaluateMetricType.Default;
        }

        internal LightGbmBinaryTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName))
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmBinaryTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of The label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
        internal LightGbmBinaryTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Options.Defaults.NumberOfIterations)
            : base(env, LoadNameValue, TrainerUtils.MakeBoolScalarLabel(labelColumnName), featureColumnName, exampleWeightColumnName, null, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations)
        {
        }

        private protected override CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator> CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null, "The predictor cannot be created before training is complete");
            var innerArgs = LightGbmInterfaceUtils.JoinParameters(Options);
            var pred = new LightGbmBinaryModelParameters(Host, TrainedEnsemble, FeatureCount, innerArgs);
            var cali = new PlattCalibrator(Host, -0.5, 0);
            return new FeatureWeightsCalibratedModelParameters<LightGbmBinaryModelParameters, PlattCalibrator>(Host, pred, cali);
        }

        private protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            var labelType = data.Schema.Label.Value.Type;
            if (!(labelType is BooleanDataViewType || labelType is KeyType || labelType == NumberDataViewType.Single))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Value.Name}' is of type '{labelType.RawType}', but must be unsigned int, boolean or float.");
            }
        }

        private protected override void GetDefaultParameters(IChannel ch, int numRow, bool hasCategorical, int totalCats, bool hiddenMsg = false)
        {
            base.GetDefaultParameters(ch, numRow, hasCategorical, totalCats, true);
            LightGbmTrainerOptions.EvaluationMetric = Options.EvaluateMetricType.Logloss;
            var internalOptions = (InternalOptions)LightGbmTrainerOptions;
            internalOptions.Objective = "binary";
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>
            MakeTransformer(CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator> model, DataViewSchema trainSchema)
         => new BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="LightGbmBinaryTrainer"/> using both training and validation data, returns
        /// a <see cref="BinaryPredictionTransformer{CalibratedModelParametersBase}"/>.
        /// </summary>
        public BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);
    }

    /// <summary>
    /// A component to train a LightGBM model.
    /// </summary>
    internal static partial class LightGbm
    {
        [TlcModule.EntryPoint(
            Name = "Trainers.LightGbmBinaryClassifier",
            Desc = LightGbmBinaryTrainer.Summary,
            UserName = LightGbmBinaryTrainer.UserName,
            ShortName = LightGbmBinaryTrainer.ShortName)]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, LightGbmBinaryTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<LightGbmBinaryTrainer.Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LightGbmBinaryTrainer(host, input),
                getLabel: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                getWeight: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName));
        }
    }
}
