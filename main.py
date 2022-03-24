from matplotlib import pyplot
from numpy import linalg
from sklearn.cluster import KMeans
import csv
import numpy
import pandas
import sys


# This function will read in the csv file specified in the command line (if the file is valid),
# and provides hints to the user if there is anything wrong with the command line arguments.
if __name__ == '__main__':
    # If the amount of arguments (plus the name of the program and the amount of records to analyze) is not 3, we will
    # inform the user.
    if len(sys.argv) != 3:
        print("Error - invalid number of arguments (must specify the csv file and the number of records to analyze)")
        exit()

    # We want to retrieve the number of columns that we have to analyze (excluding the ID)
    with open(sys.argv[1]) as csv_file:
        read_data = csv.reader(csv_file)
        headers = next(read_data)[1:]
        number_col = len(headers)  # The total number of columns in the file (minus the ID)
        total_records = len(csv_file.readlines())  # The total number of records in the file (minus the header)
    csv_file.close()

    # If the specified number of records to analyze is less than the number of columns in the csv file (excluding
    # the ID), we inform the user.
    if int(sys.argv[2]) < number_col:
        print("Error - the number of records to analyze must be greater than", number_col)
        exit()

    # If the specified number of records to analyze is greater than the number of records there actually is
    # (excluding the ID), we inform the user.
    if int(sys.argv[2]) > total_records:
        print("Error - the number of records to analyze must not be greater than", total_records)

    else:
        try:
            # The second cmd argument is the csv file we have to open and retrieve data from
            with open(sys.argv[1]) as csv_file:
                read_data = csv.reader(csv_file)
                headers = next(read_data)[1:]  # A list of the names of the item groups (excluding ID)
                number_col = len(headers)  # Find the amount of item groups there are
                item_lists = [[]] * number_col  # An array of the frequencies for each item group
                num_records = int(sys.argv[2])  # The number of records to analyze

                # For each row of data we append individual values to its corresponding group column
                for row in read_data:
                    # If we analyzed enough records, we exit this loop
                    if num_records == 0:
                        break

                    item_counter = 0

                    # We want to skip over the IDs
                    for row_item in row[1:]:
                        frequency = int(row_item.strip())

                        # If the array is empty, we initialize one with the value to append included
                        if not item_lists[item_counter]:
                            item_lists[item_counter] = [frequency]
                        else:
                            item_lists[item_counter].append(frequency)
                        item_counter += 1

                    num_records -= 1

                dictionary_data = dict(zip(headers, zip(*item_lists)))  # Converts item_lists to a dictionary
                data_frame = pandas.DataFrame(dictionary_data, columns=headers)  # Convert the dictionary to data frame
                correlation_matrix = data_frame.cov()  # We get a covariance matrix
                eig_values, eig_vectors = linalg.eig(correlation_matrix)  # Retrieve the eigenvalues and eigenvectors

                # Put eigenvalues into a python list for easy removal
                new_eig_values = []
                eig_total = 0
                for value in eig_values:
                    eig_total += value
                    new_eig_values.append(value)

                # Put eigenvectors into a python list for easy removal
                new_eig_vectors = []
                for vector in eig_vectors:
                    new_eig_vectors.append(vector)

                # Determine the sum of all the eigenvalues
                eig_total = 0
                for value in new_eig_values:
                    eig_total += value

                normal_values = []  # The list of normalized and sorted eigenvalues
                sorted_vectors = []  # The list of sorted eigenvectors

                # Loop until the values are all in the new value list
                while new_eig_values:
                    best_index = 0
                    best_value = 0
                    current_index = 0

                    # We try to determine the largest value in the value list
                    for value in new_eig_values:
                        if value > best_value:
                            best_value = value
                            best_index = current_index
                        current_index += 1
                    normal_values.append(new_eig_values[best_index] / eig_total)  # Append normalized max value
                    new_eig_values.remove(best_value)  # Remove value from old list
                    sorted_vectors.append(new_eig_vectors[best_index])  # Append vector corresponding to max value
                    new_eig_vectors.pop(best_index)  # Remove vector from old list

                cum_sums = numpy.array(normal_values).cumsum()  # Get the cumulative sum of the normalized values
                pyplot.title("Cumulative Sum of Normalized Eigenvalues")
                pyplot.plot(cum_sums)
                pyplot.show()  # Show the cumulative sum plot

                # Prints out the first three eigenvectors
                print("First 3 eigenvectors:")
                print(sorted_vectors[0])
                print(sorted_vectors[1])
                print(sorted_vectors[2])

                # Project csv data onto first two eigenvectors, then show scatter plot
                two_eig = [sorted_vectors[0], sorted_vectors[1]]
                projected_data = numpy.dot(two_eig, item_lists)
                pyplot.scatter(projected_data[0], projected_data[1])
                pyplot.title("Projection of Agglomeration data onto first two eigenvectors")
                pyplot.xlabel('X-Values')
                pyplot.ylabel('Y-Values')
                pyplot.show()

                # We utilize the 'Knee' method from K-means to determine the optimal amount of clusters
                sum_square = []
                for i in range(1, 11):
                    k_means = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                    k_means.fit(projected_data.reshape(-1, 1))
                    sum_square.append(k_means.inertia_)
                pyplot.plot(range(1, 11), sum_square)
                pyplot.title('Elbow Method')
                pyplot.xlabel('Number of clusters')
                pyplot.ylabel('Within-Cluster Sum of Square')
                pyplot.show()

                # We perform k-means on the optimal amount of clusters (in this case, it was 4)
                k_means = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
                pred_y = k_means.fit_predict(projected_data.reshape(-2, 2))
                pyplot.scatter(projected_data[0], projected_data[1])
                pyplot.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=300, c='red')
                pyplot.title("Re-projection of Cluster COMs onto first two eigenvectors")
                pyplot.xlabel('X-Values')
                pyplot.ylabel('Y-Values')
                pyplot.show()

                # Print out the center of mass for all the clusters
                print()
                print("Cluster Centers for K-means:")
                for com in k_means.cluster_centers_:
                    print(com)

                # Print out the vector you get after re-projection
                print()
                print("Re-projected values:")
                re_project = projected_data = numpy.dot(k_means.cluster_centers_, two_eig)
                print(re_project)

            csv_file.close()

            # If the file is unable to be opened for whatever reason, we will inform the user.
        except OSError:
            print("Error - cannot open file '" + sys.argv[1] + "'")
