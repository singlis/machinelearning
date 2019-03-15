﻿using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.LightGbm;

[assembly: LoadableClass(typeof(GradientBooster), typeof(GradientBooster.Options),
    typeof(SignatureLightGBMBooster), GradientBooster.FriendlyName, GradientBooster.Name)]
[assembly: LoadableClass(typeof(DartBooster), typeof(DartBooster.Options),
    typeof(SignatureLightGBMBooster), DartBooster.FriendlyName, DartBooster.Name)]
[assembly: LoadableClass(typeof(GossBooster), typeof(GossBooster.Options),
    typeof(SignatureLightGBMBooster), GossBooster.FriendlyName, GossBooster.Name)]

[assembly: EntryPointModule(typeof(GradientBooster.Options))]
[assembly: EntryPointModule(typeof(DartBooster.Options))]
[assembly: EntryPointModule(typeof(GossBooster.Options))]

namespace Microsoft.ML.Trainers.LightGbm
{
    internal delegate void SignatureLightGBMBooster();

    [TlcModule.ComponentKind("BoosterParameterFunction")]
    internal interface IBoosterParameterFactory : IComponentFactory<BoosterParameterBase>
    {
        new BoosterParameterBase CreateComponent(IHostEnvironment env);
    }

    public abstract class BoosterParameterBase
    {
        private protected static Dictionary<string, string> NameMapping = new Dictionary<string, string>()
        {
           {nameof(OptionsBase.MinimumSplitGain),               "min_split_gain" },
           {nameof(OptionsBase.MaximumTreeDepth),               "max_depth"},
           {nameof(OptionsBase.MinimumChildWeight),             "min_child_weight"},
           {nameof(OptionsBase.SubsampleFraction),              "subsample"},
           {nameof(OptionsBase.SubsampleFrequency),             "subsample_freq"},
           {nameof(OptionsBase.L1Regularization),               "reg_alpha"},
           {nameof(OptionsBase.L2Regularization),               "reg_lambda"},
           {nameof(OptionsBase.WeightOfPositiveExamples),       "scale_pos_weight"}
        };

        public abstract class OptionsBase : IBoosterParameterFactory
        {
            internal BoosterParameterBase GetBooster() { return null; }

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use for binary classification when training data is not balanced.", ShortName = "us")] public bool UnbalancedSets = false;

            /// <summary>
            /// The minimum loss reduction required to make a further partition on a leaf node of the tree.
            /// </summary>
            /// <value>
            /// Larger values make the algorithm more conservative.
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, " +
                    "the more conservative the algorithm will be.")]
            [TlcModule.Range(Min = 0.0)]
            public double MinimumSplitGain = 0;

            /// <summary>
            /// The maximum depth of a tree.
            /// </summary>
            /// <value>
            /// 0 means no limit.
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum depth of a tree. 0 means no limit. However, tree still grows by best-first.")]
            [TlcModule.Range(Min = 0, Max = int.MaxValue)]
            public int MaximumTreeDepth = 0;

            /// <summary>
            /// The minimum sum of instance weight needed to form a new node.
            /// </summary>
            /// <value>
            /// If the tree partition step results in a leaf node with the sum of instance weight less than <see cref="MinimumChildWeight"/>,
            /// the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number
            /// of instances needed to be in each node. The larger, the more conservative the algorithm will be.
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf " +
                    "node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, " +
                    "this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.")]
            [TlcModule.Range(Min = 0.0)]
            public double MinimumChildWeight = 0.1;

            /// <summary>
            /// The frequency of performing subsampling (bagging).
            /// </summary>
            /// <value>
            /// 0 means disable bagging. N means perform bagging at every N iterations.
            /// To enable bagging, <see cref="SubsampleFraction"/> should also be set to a value less than 1.0.
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Subsample frequency for bagging. 0 means no subsample. "
                + "Specifies the frequency at which the bagging occurs, where if this is set to N, the subsampling will happen at every N iterations." +
                "This must be set with Subsample as this specifies the amount to subsample.")]
            [TlcModule.Range(Min = 0, Max = int.MaxValue)]
            public int SubsampleFrequency = 0;

            /// <summary>
            /// The fraction of training data used for creating trees.
            /// </summary>
            /// <value>
            /// Setting it to 0.5 means that LightGBM randomly picks half of the data points to grow trees.
            /// This can be used to speed up training and to reduce over-fitting. Valid range is (0,1].
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Subsample ratio of the training instance. Setting it to 0.5 means that LightGBM randomly collected " +
                    "half of the data instances to grow trees and this will prevent overfitting. Range: (0,1].")]
            [TlcModule.Range(Inf = 0.0, Max = 1.0)]
            public double SubsampleFraction = 1;

            /// <summary>
            /// The fraction of features used when creating trees.
            /// </summary>
            /// <value>
            /// If <see cref="FeatureFraction"/> is smaller than 1.0, LightGBM will randomly select fraction of features to train each tree.
            /// For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree.
            /// This can be used to speed up training and to reduce over-fitting. Valid range is (0,1].
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Subsample ratio of columns when constructing each tree. Range: (0,1].",
                ShortName = "ff")]
            [TlcModule.Range(Inf = 0.0, Max = 1.0)]
            public double FeatureFraction = 1;

            /// <summary>
            /// The L2 regularization term on weights.
            /// </summary>
            /// <value>
            /// Increasing this value could help reduce over-fitting.
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "L2 regularization term on weights, increasing this value will make model more conservative.",
                ShortName = "l2")]
            [TlcModule.Range(Min = 0.0)]
            [TGUI(Label = "Lambda(L2)", SuggestedSweeps = "0,0.5,1")]
            [TlcModule.SweepableDiscreteParam("RegLambda", new object[] { 0f, 0.5f, 1f })]
            public double L2Regularization = 0.01;

            /// <summary>
            /// The L1 regularization term on weights.
            /// </summary>
            /// <value>
            /// Increasing this value could help reduce over-fitting.
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "L1 regularization term on weights, increase this value will make model more conservative.",
                ShortName = "l1")]
            [TlcModule.Range(Min = 0.0)]
            [TGUI(Label = "Alpha(L1)", SuggestedSweeps = "0,0.5,1")]
            [TlcModule.SweepableDiscreteParam("RegAlpha", new object[] { 0f, 0.5f, 1f })]
            public double L1Regularization = 0;

            /// <summary>
            /// Controls the balance of positive and negative weights in <see cref="LightGbmBinaryClassificationTrainer"/>.
            /// </summary>
            /// <value>
            /// This is useful for training on unbalanced data. A typical value to consider is sum(negative cases) / sum(positive cases).
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Control the balance of positive and negative weights, useful for unbalanced classes." +
                " A typical value to consider: sum(negative cases) / sum(positive cases).",
                ShortName = "ScalePosWeight")]
            public double WeightOfPositiveExamples = 1;

            BoosterParameterBase IComponentFactory<BoosterParameterBase>.CreateComponent(IHostEnvironment env)
            {
                return BuildOptions();
            }

            BoosterParameterBase IBoosterParameterFactory.CreateComponent(IHostEnvironment env)
            {
                return BuildOptions();
            }

            internal abstract BoosterParameterBase BuildOptions();

        }

        internal void UpdateParameters(Dictionary<string, object> res)
        {
            FieldInfo[] fields = BoosterOptions.GetType().GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            foreach (var field in fields)
            {
                var attribute = field.GetCustomAttribute<ArgumentAttribute>(false);

                if (attribute == null)
                    continue;

                var name = NameMapping.ContainsKey(field.Name) ? NameMapping[field.Name] : LightGbmInterfaceUtils.GetOptionName(field.Name);
                res[name] = field.GetValue(BoosterOptions);
            }
        }

        /// <summary>
        /// Create <see cref="IBoosterParameterFactory"/> for supporting legacy infra built upon <see cref="IComponentFactory"/>.
        /// </summary>
        internal abstract IBoosterParameterFactory BuildFactory();
        internal abstract string BoosterName { get; }

        private protected OptionsBase BoosterOptions;
    }

    public sealed class GradientBooster : BoosterParameterBase
    {
        //internal const string Name = "gbdt";
        internal const string FriendlyName = "Tree Booster";
        internal const string Name = "gbdt";

        [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Traditional Gradient Boosting Decision Tree.")]
        public sealed class Options : OptionsBase
        {
            internal override BoosterParameterBase BuildOptions() => new GradientBooster(this);
        }

        public GradientBooster(Options options)
        {
            Contracts.CheckUserArg(options.MinimumSplitGain >= 0, nameof(Options.MinimumSplitGain), "must be >= 0.");
            Contracts.CheckUserArg(options.MinimumChildWeight >= 0, nameof(Options.MinimumChildWeight), "must be >= 0.");
            Contracts.CheckUserArg(options.SubsampleFraction > 0 && options.SubsampleFraction <= 1, nameof(Options.SubsampleFraction), "must be in (0,1].");
            Contracts.CheckUserArg(options.FeatureFraction > 0 && options.FeatureFraction <= 1, nameof(Options.FeatureFraction), "must be in (0,1].");
            Contracts.CheckUserArg(options.L2Regularization >= 0, nameof(Options.L2Regularization), "must be >= 0.");
            Contracts.CheckUserArg(options.L1Regularization >= 0, nameof(Options.L1Regularization), "must be >= 0.");
            Contracts.CheckUserArg(options.WeightOfPositiveExamples > 0, nameof(Options.WeightOfPositiveExamples), "must be >= 0.");
            BoosterOptions = options;
        }

        internal override IBoosterParameterFactory BuildFactory() => BoosterOptions;

        internal override string BoosterName => Name;
    }

    public sealed class DartBooster : BoosterParameterBase
    {
        internal const string Name = "dart";
        internal const string FriendlyName = "Tree Dropout Tree Booster";

        [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Dropouts meet Multiple Additive Regresion Trees. See https://arxiv.org/abs/1505.01866")]
        public sealed class Options : OptionsBase
        {
            static Options()
            {
                // Add additional name mappings
                NameMapping.Add(nameof(TreeDropFraction),                      "drop_rate");
                NameMapping.Add(nameof(MaximumNumberOfDroppedTreesPerRound),   "max_drop");
                NameMapping.Add(nameof(SkipDropFraction),                      "skip_drop");
            }

            /// <summary>
            /// The dropout rate, i.e. the fraction of previous trees to drop during the dropout.
            /// </summary>
            /// <value>
            /// Valid range is [0,1].
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The drop ratio for trees. Range:(0,1).")]
            [TlcModule.Range(Inf = 0.0, Max = 1.0)]
            public double TreeDropFraction = 0.1;

            /// <summary>
            /// The maximum number of dropped trees in a boosting round.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of dropped trees in a boosting round.")]
            [TlcModule.Range(Inf = 0, Max = int.MaxValue)]
            public int MaximumNumberOfDroppedTreesPerRound = 1;

            /// <summary>
            /// The probability of skipping the dropout procedure during a boosting iteration.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Probability for not dropping in a boosting round.")]
            [TlcModule.Range(Inf = 0.0, Max = 1.0)]
            public double SkipDropFraction = 0.5;

            /// <summary>
            /// Whether to enable xgboost dart mode.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "True will enable xgboost dart mode.")]
            public bool XgboostDartMode = false;

            /// <summary>
            /// Whether to enable uniform drop.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "True will enable uniform drop.")]
            public bool UniformDrop = false;

            internal override BoosterParameterBase BuildOptions() => new DartBooster(this);
        }

        internal DartBooster(Options options)
        {
            Contracts.CheckUserArg(options.TreeDropFraction > 0 && options.TreeDropFraction < 1, nameof(options.TreeDropFraction), "must be in (0,1).");
            Contracts.CheckUserArg(options.SkipDropFraction >= 0 && options.SkipDropFraction < 1, nameof(options.SkipDropFraction), "must be in [0,1).");
            BoosterOptions = options;
        }

        internal override IBoosterParameterFactory BuildFactory() => BoosterOptions;
        internal override string BoosterName => Name;
    }

    /// <summary>
    /// Gradient-based One-Side Sampling booster.
    /// </summary>
    /// <remarks>
    /// For details, please see <a href="https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf">here</a>.
    /// </remarks>
    public sealed class GossBooster : BoosterParameterBase
    {
        internal const string Name = "goss";
        internal const string FriendlyName = "Gradient-based One-Size Sampling";

        [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Gradient-based One-Side Sampling.")]
        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// The retain ratio of large gradient data.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Retain ratio for large gradient instances.")]
            [TlcModule.Range(Inf = 0.0, Max = 1.0)]
            public double TopRate = 0.2;

            /// <summary>
            /// The retain ratio of small gradient data.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Retain ratio for small gradient instances.")]
            [TlcModule.Range(Inf = 0.0, Max = 1.0)]
            public double OtherRate = 0.1;

            internal override BoosterParameterBase BuildOptions() => new GossBooster(this);
        }

        internal GossBooster(Options options)
        {
            Contracts.CheckUserArg(options.TopRate > 0 && options.TopRate < 1, nameof(Options.TopRate), "must be in (0,1).");
            Contracts.CheckUserArg(options.OtherRate >= 0 && options.OtherRate < 1, nameof(Options.TopRate), "must be in [0,1).");
            Contracts.Check(options.TopRate + options.OtherRate <= 1, "Sum of topRate and otherRate cannot be larger than 1.");
            BoosterOptions = options;
        }

        internal override IBoosterParameterFactory BuildFactory() => BoosterOptions;
        internal override string BoosterName => Name;
    }
}