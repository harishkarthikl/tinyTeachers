 function readURL(input) {
    if (input.files && input.files[0]) {
                var reader = new FileReader();

    var img = document.getElementById("usr_img")
        reader.onload = function (e) {
            img.src = e.target.result;
        };

    reader.readAsDataURL(input.files[0]);

    }
 }


$( document ).ready(function() {

    $('#loader').hide();

    $('#snap_class_A_btn').click(function() {
        window.location.href='/capture?pred_class=ClassA';
    });

    $('#snap_class_B_btn').click(function() {
        window.location.href='/capture?pred_class=ClassB';
    });

    $('#snap_class_C_btn').click(function() {
        window.location.href='/capture?pred_class=ClassC';
    });

    $('#see_train_btn').click(function() {
        window.location.href='/see_train';
    });

    $('#see_pred_btn').click(function() {
        window.location.href='/see_predict';
    });

    $('#see_return_btn').click(function() {
        window.location.href='/see';
    });

    $('#upload').change(function() {
        readURL(this);
    });

    $('#predict').click(function() {
        $('#loader').show();

        var form = $('#upload_form')[0];
        var formdata = new FormData(form);

        $.ajax('/predict', {
            type: 'POST',
            data: formdata,
            contentType: false,
            processData: false,
            async: true,
            cache: false,
            success: function(data) {
                $('#loader').hide();
                $('#pred').text(data);
                $('#pred').show();
            }
        })
    });

//    $('#uptry').change(function() {
//        window.location.href='/uptry';
//    });

});