import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualization_by_dynamic(d1, d2, d3, f1, f2, f3, args, fig_suffix=''):
    """Using PCA or tSNE for generated and original data visualization.

  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """
    if fig_suffix != '':
        fig_suffix = '_' + fig_suffix
    # Analysis sample size (for faster computation)
    # anal_sample_no = min([3000, max(len(ori_data)])
    # idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    # pick at most 1000 samples for each d1, d2, d3,f1,f2,f3
    # the the shortest len of d1, d2, d3,f1,f2,f3 as l

    l = min([d1.shape[0], d2.shape[0], d3.shape[0], f1.shape[0], f2.shape[0], f3.shape[0]])
    # # pick at most max(l,1000) samples
    anal_sample_no = min(l, 2000)
    np.random.seed(args.seed)
    idx = np.random.permutation(l)[:anal_sample_no]
    #
    # # slice each d1, d2, d3,f1,f2,f3 with idx
    d1 = d1[idx]
    d2 = d2[idx]
    d3 = d3[idx]
    f1 = f1[idx]
    f2 = f2[idx]
    f3 = f3[idx]

    # Data preprocessing
    seq_len = args.max_seq_len
    print('d1', d1.shape)
    for i in range(d1.shape[0]):
        if (i == 0):

            d1_prep_data = np.reshape(np.mean(d1[0, :, :], 1), [1, seq_len])
        else:

            d1_prep_data = np.concatenate((d1_prep_data,
                                           np.reshape(np.mean(d1[i, :, :], 1), [1, seq_len])))
    for i in range(d2.shape[0]):
        if (i == 0):

            d2_prep_data = np.reshape(np.mean(d2[0, :, :], 1), [1, seq_len])
        else:

            d2_prep_data = np.concatenate((d2_prep_data,
                                           np.reshape(np.mean(d2[i, :, :], 1), [1, seq_len])))
    for i in range(d3.shape[0]):
        if (i == 0):

            d3_prep_data = np.reshape(np.mean(d3[0, :, :], 1), [1, seq_len])
        else:

            d3_prep_data = np.concatenate((d3_prep_data,
                                           np.reshape(np.mean(d3[i, :, :], 1), [1, seq_len])))
    # do the same to f1,f2,f3
    for i in range(f1.shape[0]):
        if (i == 0):

            f1_prep_data = np.reshape(np.mean(f1[0, :, :], 1), [1, seq_len])
        else:

            f1_prep_data = np.concatenate((f1_prep_data,
                                           np.reshape(np.mean(f1[i, :, :], 1), [1, seq_len])))
    for i in range(f2.shape[0]):
        if (i == 0):

            f2_prep_data = np.reshape(np.mean(f2[0, :, :], 1), [1, seq_len])
        else:

            f2_prep_data = np.concatenate((f2_prep_data,
                                           np.reshape(np.mean(f2[i, :, :], 1), [1, seq_len])))
    for i in range(f3.shape[0]):
        if (i == 0):

            f3_prep_data = np.reshape(np.mean(f3[0, :, :], 1), [1, seq_len])
        else:

            f3_prep_data = np.concatenate((f3_prep_data,
                                           np.reshape(np.mean(f3[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["tab:blue" for i in range(d1.shape[0])] + ["tab:orange" for i in range(d2.shape[0])] + ["tab:green" for i
                                                                                                      in range(
            d3.shape[0])] + \
             ["tab:red" for i in range(f1.shape[0])] + ["tab:purple" for i in range(f2.shape[0])] + ["tab:brown" for i
                                                                                                     in
                                                                                                     range(f3.shape[0])]

    # Do t-SNE Analysis together
    prep_data_final = np.concatenate((d1_prep_data, d2_prep_data, d3_prep_data, f1_prep_data, f2_prep_data, f3_prep_data
                                      ), axis=0)

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=6000, random_state=args.seed)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plotting
    # increase the size of the figure
    plt.figure(figsize=(20, 20))
    f, ax = plt.subplots(1)

    plt.scatter(
        tsne_results[d1.shape[0] + d2.shape[0] + d3.shape[0]:d1.shape[0] + d2.shape[0] + d3.shape[0] + f1.shape[0], 0],
        tsne_results[d1.shape[0] + d2.shape[0] + d3.shape[0]:d1.shape[0] + d2.shape[0] + d3.shape[0] + f1.shape[0], 1],
        c='blue', alpha=0.5, marker='o', label="Dynamics 0", s=4)
    plt.scatter(tsne_results[
                d1.shape[0] + d2.shape[0] + d3.shape[0] + f1.shape[0]:d1.shape[0] + d2.shape[0] + d3.shape[0] +
                                                                      f1.shape[0] + f2.shape[0], 0],
                tsne_results[
                d1.shape[0] + d2.shape[0] + d3.shape[0] + f1.shape[0]:d1.shape[0] + d2.shape[0] + d3.shape[0] +
                                                                      f1.shape[0] + f2.shape[0], 1],
                c='green', alpha=0.5, marker='o', label="Dynamics 1", s=4)
    plt.scatter(tsne_results[d1.shape[0] + d2.shape[0] + d3.shape[0] + f1.shape[0] + f2.shape[0]:, 0],
                tsne_results[d1.shape[0] + d2.shape[0] + d3.shape[0] + f1.shape[0] + f2.shape[0]:, 1],
                c='red', alpha=0.5, marker='p', label="Dynamics 2", s=4)

    ax.legend()
    ax.legend().set_visible(False)
    plt.axis('off')

    plt.savefig(args.model_path + f'/tsne{fig_suffix}.png', dpi=600)
    print('figure saved to', args.model_path + f'/tsne{fig_suffix}.png')


def visualization_by_dynamic_solo(d1, d2, d3, args, fig_suffix=''):
    """Using PCA or tSNE for generated and original data visualization.

  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """
    print('d1', d1.shape)
    print('d2', d2.shape)
    print('d3', d3.shape)
    if fig_suffix != '':
        fig_suffix = '_' + fig_suffix

    l = min([d1.shape[0], d2.shape[0], d3.shape[0]])
    # # pick at most max(l,1000) samples
    anal_sample_no = min(l, 2000)
    np.random.seed(args.seed)
    idx = np.random.permutation(l)[:anal_sample_no]
    #
    # # slice each d1, d2, d3,f1,f2,f3 with idx
    d1 = d1[idx]
    d2 = d2[idx]
    d3 = d3[idx]

    seq_len = args.max_seq_len
    # print('d1',d1.shape)
    for i in range(d1.shape[0]):
        if (i == 0):

            d1_prep_data = np.reshape(np.mean(d1[0, :, :], 1), [1, seq_len])
        else:
            d1_prep_data = np.concatenate((d1_prep_data,
                                           np.reshape(np.mean(d1[i, :, :], 1), [1, seq_len])))
    for i in range(d2.shape[0]):
        if (i == 0):
            d2_prep_data = np.reshape(np.mean(d2[0, :, :], 1), [1, seq_len])
        else:
            d2_prep_data = np.concatenate((d2_prep_data,
                                           np.reshape(np.mean(d2[i, :, :], 1), [1, seq_len])))
    for i in range(d3.shape[0]):
        if (i == 0):
            d3_prep_data = np.reshape(np.mean(d3[0, :, :], 1), [1, seq_len])
        else:
            d3_prep_data = np.concatenate((d3_prep_data,
                                           np.reshape(np.mean(d3[i, :, :], 1), [1, seq_len])))

    prep_data_final = np.concatenate((d1_prep_data, d2_prep_data, d3_prep_data), axis=0)

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=6000, random_state=args.seed)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plotting
    plt.figure(figsize=(20, 20))
    f, ax = plt.subplots(1)

    plt.scatter(tsne_results[:d1.shape[0], 0], tsne_results[:d1.shape[0], 1],
                c='blue', alpha=0.5, marker='o', label="Dynamics 0", s=4)
    plt.scatter(tsne_results[d1.shape[0]:d1.shape[0] + d2.shape[0], 0],
                tsne_results[d1.shape[0]:d1.shape[0] + d2.shape[0], 1],
                c='green', alpha=0.5, marker='o', label="Dynamics 1", s=4)
    plt.scatter(tsne_results[d1.shape[0] + d2.shape[0]:, 0], tsne_results[d1.shape[0] + d2.shape[0]:, 1],
                c='red', alpha=0.5, marker='o', label="Dynamics 2", s=4)

    ax.legend()

    ax.legend().set_visible(False)
    plt.axis('off')
    plt.savefig(args.model_path + f'/tsne{fig_suffix}.png', dpi=600)
    print('figure saved to', args.model_path + f'/tsne{fig_suffix}.png')


def visualization(generated_data, ori_data, analysis, args, fig_suffix=''):
    """Using PCA or tSNE for generated and original data visualization.

  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """
    if fig_suffix != '':
        fig_suffix = '_' + fig_suffix
    # Analysis sample size (for faster computation)
    anal_sample_no = min([2000, len(ori_data)])
    np.random.seed(args.seed)
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape
    # print('no, seq_len, dim', no, seq_len, dim)
    plt.figure(figsize=(20, 20))

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["tab:blue" for i in range(anal_sample_no)] + ["tab:orange" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2, random_state=args.seed)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.5, label="Real", s=4)
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.5, label="Generated", s=4)

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        # plt.show()
        # savefig to file
        plt.savefig(args.model_path + f'/pca{fig_suffix}.png')

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=6000, random_state=args.seed)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.5, label="Real", s=4)
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.5, label="Generated", s=4)

        ax.legend()
        ax.legend().set_visible(False)
        plt.axis('off')
        plt.savefig(args.model_path + f'/tsne{fig_suffix}.png', dpi=600)
        print('figure saved to', args.model_path + f'/tsne{fig_suffix}.png')
