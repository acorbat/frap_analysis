import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm


def load_df(path):
    """loads a Pandas DataFrame from path"""
    return pd.read_pickle(str(path))


def exponential(t, amplitude, characteristic_time, constant):
    """Exponential function to fit FRAP recovery curves.

    Parameters
    ----------
    t : list, numpy.array
        timepoints of function evaluation
    amplitude : float
        Amplitude of recovery through time
    characteristic_time : float
        Characteristic time of FRAP recovery
    constant : float
        Immobile fraction

    Returns
    -------
    Depending on input, returns value, list or array of FRAP recovery for the input timepoints
    """
    return 1 - amplitude * np.exp(-t / characteristic_time) - constant


def fit_simple_exponential(x, y):
    """Takes two vector like elements and fits them with a single exponential function.

    Parameters
    ----------
    x: list, np.array
        Values taken by the independent variable
    y: list, np.array
        Values taken by the dependent variable

    Returns
    -------
    : lmfit.ModelResult class
        Returns a class containing the best fit parameters and other statistical estimators of goodness of fit.

    """
    # Model Instantiation
    recovery_model = lm.Model(exponential, independent_vars=['t'])
    recovery_params = recovery_model.make_params(amplitude=0.7,
                                        characteristic_time=50,
                                        constant=0.3)
    recovery_params['amplitude'].set(min=0)
    recovery_params['characteristic_time'].set(min=0)
    recovery_params['constant'].set(min=0)

    return recovery_model.fit(y, params=recovery_params, t=x)


def fit_df_simple_exponential(df, columns=['date', 'condition', 'experiment', 'cell']):
    """Applies simple exponential fit to every row of DataFrame df of kind "pos" and grouping them by columns."""
    df['amplitude'] = np.nan
    df['characteristic_time'] = np.nan
    df['immobile_fraction'] = np.nan

    df_pos = df.query('kind == "pos"')

    for column_values, this_df in df_pos.groupby(columns):
        for particle, particle_df in this_df.groupby('particle'):
            if len(particle_df) > 1:
                raise Exception('More than one element in grouped by DataFrame in %s, %s, %s, %s.' % column_values)
            time = particle_df.time.values[0]
            intensity = particle_df.intensity.values[0]
            index = particle_df.index.values[0]

            if len(time) <= 3 or np.isnan(intensity).all():
                continue

            result = fit_simple_exponential(time, intensity)

            df.at[index, 'amplitude'] = result.best_values['amplitude']
            df.at[index, 'characteristic_time'] = result.best_values['characteristic_time']
            df.at[index, 'immobile_fraction'] = result.best_values['constant']

    return df


def double_exponential(t, amplitude_1, characteristic_time_1, amplitude_2, characteristic_time_2, constant):
    """Exponential function to fit FRAP recovery curves.

    Parameters
    ----------
    t : list, numpy.array
        timepoints of function evaluation
    amplitude_1 : float
        Amplitude of first component of recovery through time
    characteristic_time_1 : float
        Characteristic time of first component of FRAP recovery
    amplitude_1 : float
        Amplitude of second component of recovery through time
    characteristic_time_1 : float
        Characteristic time of second component of FRAP recovery
    constant : float
        Immobile fraction

    Returns
    -------
    Depending on input, returns value, list or array of FRAP recovery for the input timepoints
    """
    return 1 - amplitude_1 * np.exp(-t / characteristic_time_1) - \
           amplitude_2 * np.exp(-t / characteristic_time_2) - \
           constant


def fit_double_exponential(x, y):
    """Takes two vector like elements and fits them with a double exponential function.

        Parameters
        ----------
        x: list, np.array
            Values taken by the independent variable
        y: list, np.array
            Values taken by the dependent variable

        Returns
        -------
        : lmfit.ModelResult class
            Returns a class containing the best fit parameters and other statistical estimators of goodness of fit.

        """
    # Model Instantiation
    recovery_model = lm.Model(double_exponential, independent_vars=['t'])
    recovery_params = recovery_model.make_params(amplitude_1=0.35,
                                                 characteristic_time_1=80,
                                                 amplitude_2=0.35,
                                                 characteristic_time_2=10,
                                                 constant=0.3)
    recovery_params['amplitude_1'].set(min=0)
    recovery_params['characteristic_time_1'].set(min=0)
    recovery_params['amplitude_2'].set(min=0)
    recovery_params['characteristic_time_2'].set(min=0)
    recovery_params['constant'].set(min=0)

    result = recovery_model.fit(y, params=recovery_params, t=x)

    # TODO: this doesn't correct the other measurements of the fit.
    if result.best_values['characteristic_time_1'] < result.best_values['characteristic_time_2']:
        result.best_values['char_time'] = result.best_values.pop('characteristic_time_2')
        result.best_values['characteristic_time_2'] = result.best_values.pop('characteristic_time_1')
        result.best_values['characteristic_time_1'] = result.best_values.pop('char_time')
        print('reordering keys in fitting result')

    return result


def fit_df_double_exponential(df, columns=['date', 'condition', 'experiment', 'cell']):
    """Applies simple exponential fit to every row of DataFrame df of kind "pos" and grouping them by columns."""
    df['amplitude_1'] = np.nan
    df['characteristic_time_1'] = np.nan
    df['amplitude_2'] = np.nan
    df['characteristic_time_2'] = np.nan
    df['immobile_fraction_double'] = np.nan

    df_pos = df.query('kind == "pos"')

    for column_values, this_df in df_pos.groupby(columns):
        for particle, particle_df in this_df.groupby('particle'):
            if len(particle_df) > 1:
                raise Exception('More than one element in grouped by DataFrame in %s, %s, %s, %s.' % column_values)
            time = particle_df.time.values[0]
            intensity = particle_df.intensity.values[0]
            index = particle_df.index.values[0]

            if len(time) <= 3 or np.isnan(intensity).all():
                continue

            result = fit_double_exponential(time, intensity)

            df.at[index, 'amplitude_1'] = result.best_values['amplitude_1']
            df.at[index, 'characteristic_time_1'] = result.best_values['characteristic_time_1']
            df.at[index, 'amplitude_2'] = result.best_values['amplitude_2']
            df.at[index, 'characteristic_time_2'] = result.best_values['characteristic_time_2']
            df.at[index, 'immobile_fraction_double'] = result.best_values['constant']

    return df
