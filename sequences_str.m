function  sequences = sequences_str( v )

    sequences = {};
    for i = v
        if ((i>=10)&&(i<=99))
            sequences = [sequences,['000',num2str(i)]];
        elseif ((i>=100)&&(i<=999))
            sequences = [sequences,['00',num2str(i)]];
        elseif ((i>=1)&&(i<=9))
            sequences = [sequences,['0000',num2str(i)]];
        else
            error('error in sequence names!!!');
        end
    end
end

