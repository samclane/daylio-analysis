import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row, widgetbox
from bokeh.models import ColumnDataSource, HBar, Line
from bokeh.models.ranges import Range1d
from bokeh.models.widgets import Slider, Button, Div, RadioButtonGroup, Toggle
from bokeh.plotting import figure

import read_and_ml as rml

Y_MIN = min(rml.MOOD_INT_TO_STR.keys())
Y_MAX = max(rml.MOOD_INT_TO_STR.keys())

# GUI
# Plot- Regression Data
SCATTER_TOOLTIPS = [("index", "$index{0}"),
                    ("(x,y)", "($x{0}, $y{0.0})"),
                    ("date", "@timestamp")]
plot_mood_scatter = figure(plot_height=400, plot_width=850,
                           title="Mood Over Time", toolbar_location="above",
                           x_axis_label='entry #', y_axis_label='mood',
                           tools="reset,pan,save,box_zoom,wheel_zoom",
                           tooltips=SCATTER_TOOLTIPS)
BAR_TOOLTIPS = [("count", "@right")]
plot_mood_bar = figure(plot_height=400, plot_width=400,
                       title="Mood Distribution", toolbar_location="above",
                       y_axis_label='mood', x_axis_label='count',
                       tools="reset,save",
                       y_range=(Y_MIN, Y_MAX), tooltips=BAR_TOOLTIPS)
# Plot Control Buttons
plot_sim = Button(label="Run")
plot_clear = Button(label="Clear")
plot_ctls = column(plot_sim, plot_clear)
# Main Control Buttons
ctl_model_title = Div(text="<h3>ML Model</h3>")
ctl_model = RadioButtonGroup(labels=['SVM', 'KNN', 'MLP', 'ARD'], active=0)
ctl_title = Div(text="<h3>Parameters</h3>")
ctl_feat_reduce = Toggle(label="Reduced Featureset")
ctl_est = Slider(title="Number of Estimators", value=rml.NUM_ESTIMATORS,
                 start=1, end=100, step=1)
ctl_pct_test = Slider(title="Percent Test", value=rml.TEST_RATIO,
                      start=.05, end=1, step=.05)
KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']
ctl_kernel = RadioButtonGroup(labels=KERNELS, active=KERNELS.index(rml.KERNEL_DEFAULT))
ctl_c_val = Slider(title="C Value", value=rml.C_VALUE,
                   start=.1, end=30, step=.1)
ctl_neighbors = Slider(title="Num Neighbors", value=rml.NUM_NEIGHBORS,
                       start=1, end=30, step=1)
ctl_num_nodes = Slider(title="Num. nodes", value=rml.NUM_NODES,
                       start=1, end=100, step=1)
ctl_hidden = Slider(title="Num. hidden layers", value=rml.NUM_HIDDEN_LAYERS,
                    start=1, end=50, step=1)
ctl_inputs = widgetbox(ctl_model_title, ctl_model, ctl_title, ctl_feat_reduce, ctl_est, ctl_pct_test, ctl_kernel,
                       ctl_c_val, ctl_neighbors, ctl_num_nodes, ctl_hidden)
# Data Sources and Initialization
d_data = rml.preprocess(rml.read_file("daylio_export.csv"))
d_features = rml.engineer_features(d_data)
if ctl_feat_reduce.active:
    d_features = rml.feature_select(d_features, d_data["mood"])
x, y = range(len(d_data)), d_data["mood"]

source_data = ColumnDataSource(data=dict(x=x, y=y, timestamp=d_data["date"] + ", " + d_data["year"].apply(str)))
pred_data = ColumnDataSource(
    data=dict(x=x, y=[0, ] * len(y), timestamp=d_data["date"] + ", " + d_data["year"].apply(str)))
plot_mood_scatter.scatter('x', 'y', source=source_data)

xrange_data = Range1d(bounds=[None, None], start=0, end=len(y))
yrange_data = Range1d(bounds=[None, None], start=Y_MIN, end=Y_MAX)

plot_mood_scatter.x_range = xrange_data
plot_mood_scatter.y_range = yrange_data

# Set up bar graph
# source_bars = d_data["mood"].value_counts()
source_bars = ColumnDataSource(dict(y=d_data["mood"].value_counts().index, right=d_data["mood"].value_counts()))
pred_line = ColumnDataSource(dict(y=d_data["mood"].value_counts().index, x=[0, ] * len(d_data["mood"].value_counts())))
hbar_glyph = HBar(y="y", right="right", left=0, height=0.3, fill_color='#1d76B4')
prebar_glyph = Line(y="y", x="x", line_color='red', line_width=3)
plot_mood_bar.add_glyph(source_bars, hbar_glyph)
plot_mood_bar.yaxis.major_label_overrides = rml.MOOD_INT_TO_STR
plot_mood_bar.ygrid.grid_line_color = None


# Callbacks
def update_plot(*args, **kwargs):
    global d_features
    # Pull params from controls
    num_est = ctl_est.value
    test_pct = ctl_pct_test.value  # No way for this to get passed
    def_kern = ctl_kernel.labels[ctl_kernel.active]
    c_val = ctl_c_val.value
    num_neigh = ctl_neighbors.value
    num_nodes = ctl_num_nodes.value
    num_hidden = ctl_hidden.value

    if ctl_feat_reduce.active:
        X_features = rml.feature_select(d_features, d_data["mood"], num_est)
    else:
        X_features = d_features

    print(X_features.keys())

    model = ctl_model.labels[ctl_model.active]
    if model == 'SVM':
        clf = rml.regress_svm(X_features, y, kernel_val=def_kern, C_val=c_val)
    elif model == "KNN":
        clf = rml.regress_knn(X_features, y, K_val=num_neigh)
    elif model == "MLP":
        clf = rml.regress_mlp(X_features, y, num_nodes, num_hidden)
    elif model == "ARD":
        clf = rml.regress_ard(X_features, y)
    else:
        raise Exception("model value not in list")

    y_pred = pd.Series(clf.predict(X_features))

    source_data.data = dict(x=x, y=y, timestamp=d_data["date"] + ", " + d_data["year"].apply(str))
    pred_data.data = dict(x=x, y=y_pred, timestamp=d_data["date"] + ", " + d_data["year"].apply(str))
    xrange_data.start = 0
    xrange_data.end = max(x)
    yrange_data.start = Y_MIN
    yrange_data.end = Y_MAX

    plot_mood_scatter.scatter('x', 'y', source=pred_data, fill_color='red', line_color=None)

    source_bars.data = dict(y=d_data["mood"].value_counts().index, right=d_data["mood"].value_counts())
    pred_line.data = dict(y=sorted(y_pred.value_counts().index), x=y_pred.value_counts().sort_index())
    plot_mood_bar.add_glyph(source_bars, hbar_glyph)
    plot_mood_bar.add_glyph(pred_line, prebar_glyph)


def clear_plot():
    source_data.data = dict(x=[], y=[])
    pred_data.data = dict(x=[], y=[])
    source_bars.data = dict(y=[], right=[])
    pred_line.data = dict(y=[], x=[])


plot_sim.on_click(update_plot)
plot_clear.on_click(clear_plot)

# Page Layout
col_inputs = column(plot_ctls, ctl_inputs)
row_plots = row(plot_mood_scatter, plot_mood_bar)
row_page = row(col_inputs, row_plots, width=1200)
curdoc().add_root(row_page)
curdoc().title = "Daylio Data Display"
