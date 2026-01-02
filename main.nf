nextflow.enable.dsl = 2

/*
 * Main workflow
 */
workflow {

    /*
     * Define ECHONEXT_DATA / RUN_DIR pairs
     * Add as many tuples as you want
     */
    data_pairs = Channel.of(
        tuple('/data/setA', '/results/setA'),
        tuple('/data/setB', '/results/setB')
    )

    data_pairs.each { pair ->
        run_pipeline(pair)
    }
}

/*
 * Per-dataset pipeline
 */
workflow run_pipeline {

    take:
    pair

    main:

    // Stage 1
    logreg(pair)
    minimodel(pair)
    resnet1d(pair)
    resnet2d(pair)

    // Stage 2
    probe_sklearn(pair)
    probe_cat1(pair)
    probe_cat3(pair)
    probe_cat4(pair)
    probe_fusion(pair)

    // Stage 3 dependency chain
    cooc = echonext_cooccurrence(pair)

    cat1 = echonext_cat1(cooc)
    cat3 = echonext_cat3(cooc)
    cat4 = echonext_cat4(cooc)

    echonext_fusion(cat1, cat3, cat4)
}

/*
 * =========================
 * Process definitions
 * =========================
 */

/* -------- CPU-only jobs -------- */

process logreg {
    label 'cpu'

    input:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/1-run-echonext-logreg.sh
    """
}

process probe_sklearn {
    label 'cpu'

    input:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/4-0-run-protoecgnet-probe-sklearn.sh
    """
}

process echonext_cooccurrence {
    label 'cpu'

    input:
    tuple val(input_data), val(output_results)

    output:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/5-0-run-protoecgnet-echonext-cooccurrence.sh
    """
}

/* -------- GPU jobs (1 GPU per job) -------- */

process minimodel {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/2-run-minimodel.sh
    """
}

process resnet1d {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/3-run-resnet-1d.sh
    """
}

process resnet2d {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/3-run-resnet-2d.sh
    """
}

process probe_cat1 {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/4-1-run-protoecgnet-probe-torch-cat1.sh
    """
}

process probe_cat3 {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/4-3-run-protoecgnet-probe-torch-cat3.sh
    """
}

process probe_cat4 {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/4-4-run-protoecgnet-probe-torch-cat4.sh
    """
}

process probe_fusion {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/4-5-run-protoecgnet-probe-torch-fusion.sh
    """
}

/* -------- Stage 5 dependency chain -------- */

process echonext_cat1 {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)

    output:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/5-1-run-protoecgnet-echonext-cat1.sh
    """
}

process echonext_cat3 {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)

    output:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/5-3-run-protoecgnet-echonext-cat3.sh
    """
}

process echonext_cat4 {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)

    output:
    tuple val(input_data), val(output_results)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/5-4-run-protoecgnet-echonext-cat4.sh
    """
}

process echonext_fusion {
    label 'gpu'
    gpu 1

    input:
    tuple val(input_data), val(output_results)
    tuple val(input_data2), val(output_results2)
    tuple val(input_data3), val(output_results3)

    script:
    """
    export ECHONEXT_DATA=${input_data}
    export RUN_DIR=${output_results}
    ./scripts/5-5-run-protoecgnet-echonext-fusion.sh
    """
}
