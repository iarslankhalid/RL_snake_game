import plotly.graph_objects as go
from IPython.display import clear_output

def plot(scores, mean_scores):
    clear_output(wait=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(scores))), y=scores, mode='lines', name='Score'))
    fig.add_trace(go.Scatter(x=list(range(len(mean_scores))), y=mean_scores, mode='lines', name='Mean Score'))
    
    fig.update_layout(title='Training...',
                      xaxis=dict(title='Number of Games'),
                      yaxis=dict(title='Score'),
                      showlegend=True)

    fig.show()
