from repeater_chain_analyzer import RepeaterChainCalculator, RepeaterChainSampler
import plotting_tools as plot_tools


def generate_data_1a(folder):
    calculator = RepeaterChainCalculator(n=3, trunc=500, pgen=0.1, pswap=0.5, outputfolder=folder)
    calculator.calculate()

def generate_data_1b(folder):
    sampler = RepeaterChainSampler(n=3, pgen=0.1, pswap=0.5, outputfolder=folder)
    sampler.sample(sample_size=100000)

def generate_data_2a(folder):
    f0 = 0.95
    w0 = (4*f0 - 1) / 3
    calculator = RepeaterChainCalculator(n=2, trunc=100, pgen=0.1, pswap=0.5, w0=w0, T_coh=50, outputfolder=folder)
    calculator.calculate()

def generate_data_2b(folder):
    f0 = 0.95
    w0 = (4*f0 - 1) / 3
    sampler = RepeaterChainSampler(n=2, pgen=0.1, pswap=0.5, w0=w0, T_coh=50, outputfolder=folder)
    sampler.sample(sample_size=250000)

def generate_data_3a(folder):
    f0 = 0.95
    w0 = (4*f0 - 1) / 3
    sampler = RepeaterChainSampler(n=2, pgen=0.1, pswap=0.5, w0=w0, T_coh=250, n_dist=1, outputfolder=folder)
    sampler.sample(sample_size=250000)

def generate_data_3b(folder):
    f0 = 0.95
    w0 = (4*f0 - 1) / 3
    sampler = RepeaterChainSampler(n=2, pgen=0.1, pswap=0.5, w0=w0, T_coh=1000, n_dist=1, outputfolder=folder)
    sampler.sample(sample_size=250000)

def generate_data_4a(folder):
    sampler = RepeaterChainSampler(n=4, pgen=0.1, pswap=0.5, comm_time=[1,2,4,8,16], outputfolder=folder)
    sampler.sample(sample_size=250000)

def generate_data_4b(folder):
    sampler = RepeaterChainSampler(n=4, pgen=0.9, pswap=0.5, comm_time=[1,2,4,8,16], outputfolder=folder)
    sampler.sample(sample_size=250000)

def load_and_plot_data_1a(folder):
    # load data
    calculator = RepeaterChainCalculator(n=0, trunc=0, pgen=0, pswap=0)
    calculator.load_from_folder(folder)

    # plot stuff
    plot_folder = folder + 'plots/'
    selection = [1,2,3]
    plot_tools.plot_distributions(plot_folder, rca=calculator, to_plot='pmf', level_selection=selection)
    plot_tools.plot_distributions(plot_folder, rca=calculator, to_plot='cdf', level_selection=selection)

def load_and_plot_data_1b(folder):
    # load data
    sampler = RepeaterChainSampler(n=0, pgen=0, pswap=0)
    sampler.load_from_folder(folder)

    # plot stuff
    plot_folder = folder + 'plots/'
    selection = [1,2,3]
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='pmf', trunc=500, level_selection=selection)
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='cdf', trunc=500, level_selection=selection)

def load_and_plot_data_2a(folder):
    # load data
    calculator = RepeaterChainCalculator(n=0, trunc=0, pgen=0, pswap=0)
    calculator.load_from_folder(folder)

    # plot stuff
    plot_folder = folder + 'plots/'
    plot_tools.plot_distributions(plot_folder, rca=calculator, to_plot='pmf')
    plot_tools.plot_distributions(plot_folder, rca=calculator, to_plot='cdf')
    plot_tools.plot_distributions(plot_folder, rca=calculator, to_plot='fid')
    plot_tools.plot_distributions(plot_folder, rca=calculator, to_plot='wern')

def load_and_plot_data_2b(folder):
    # load data
    sampler = RepeaterChainSampler(n=0, pgen=0, pswap=0)
    sampler.load_from_folder(folder)

    # plot stuff
    plot_folder = folder + 'plots/'
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='pmf', trunc=100)
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='cdf', trunc=100)
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='fid', trunc=100)
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='wern', trunc=100)

def load_and_plot_data_3(folder):
    # load data
    sampler = RepeaterChainSampler(n=0, pgen=0, pswap=0)
    sampler.load_from_folder(folder)

    # plot stuff
    plot_folder = folder + 'plots/'
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='pmf', trunc=500)
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='cdf', trunc=500)
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='fid', trunc=500)
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='wern', trunc=500)

def load_and_plot_data_4a(folder):
    # load data
    sampler = RepeaterChainSampler(n=0, pgen=0, pswap=0)
    sampler.load_from_folder(folder)

    # plot stuff
    plot_folder = folder + 'plots/'
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='pmf', trunc=2000, level_selection=[4])
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='cdf', trunc=2000, level_selection=[4])

def load_and_plot_data_4b(folder):
    # load data
    sampler = RepeaterChainSampler(n=0, pgen=0, pswap=0)
    sampler.load_from_folder(folder)

    # plot stuff
    plot_folder = folder + 'plots/'
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='pmf', trunc=500, level_selection=[4])
    plot_tools.plot_distributions(plot_folder, rca=sampler, to_plot='cdf', trunc=500, level_selection=[4])

if __name__ == '__main__':

    # Compute and plot distributions for SWAP-ONLY protocol using...
    # (a) The semi-analytical algorithm
    folder_1a = 'results/test_1a/'
    generate_data_1a(folder_1a)
    load_and_plot_data_1a(folder_1a)
    # (b) The Monte Carlo algorithm (with the same parameters)
    folder_1b = 'results/test_1b/'
    generate_data_1b(folder_1b)
    load_and_plot_data_1b(folder_1b)


    # Compute waiting time and fidelities for SWAP-ONLY protocol using...
    # (a) The semi-analytical algorithm (~15 min for trunc=100 and n=2)
    folder_2a = 'results/test_2a/'
    generate_data_2a(folder_2a)
    load_and_plot_data_2a(folder_2a)
    # (b) The Monte Carlo algorithm (with the same parameters)
    folder_2b = 'results/test_2b/'
    generate_data_2b(folder_2b)
    load_and_plot_data_2b(folder_2b)


    # Test for Monte Carlo sampling with distillation
    folder_3a = 'results/test_3a/'
    folder_3b = 'results/test_3b/'
    generate_data_3a(folder_3a)
    generate_data_3b(folder_3b)
    load_and_plot_data_3(folder_3a)
    load_and_plot_data_3(folder_3b)


    # Test for communication time
    folder_4a = 'results/test_4a/'
    folder_4b = 'results/test_4b/'
    generate_data_4a(folder_4a)
    generate_data_4b(folder_4b)
    load_and_plot_data_4a(folder_4a)
    load_and_plot_data_4b(folder_4b)
