from data_providers.cifar import Cifar10DataProvider, Cifar100DataProvider, \
	Cifar10AugmentedDataProvider, Cifar100AugmentedDataProvider


def get_data_provider_by_name(name, train_params: dict):
	""" Return required data provider class """
	if name == 'C10':
		return Cifar10DataProvider(**train_params)
	if name == 'C10+':
		return Cifar10AugmentedDataProvider(**train_params)
	if name == 'C100':
		return Cifar100DataProvider(**train_params)
	if name == 'C100+':
		return Cifar100AugmentedDataProvider(**train_params)
	else:
		print('Sorry, data provider for `%s` dataset '
		      'was not implemented yet' % name)
		exit()
