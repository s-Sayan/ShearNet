.PHONY: submit clean cleansubmit

submit:
	find unit_tests unit_test_variations -type f -name "sub.sh" \
		-exec sh -c 'cd "$$(dirname "$$1")" && bash sub.sh' _ {} \;

clean:
	find unit_tests unit_test_variations \
		\( -name "*.out" -o -name "*.err" -o -name "*.fits" -o -name "*.png" \) \
		-type f -delete

cleansubmit: clean submit