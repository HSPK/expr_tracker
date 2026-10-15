use super::*;

struct ScalarWriter(Vec<u8>);
impl Write for ScalarWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let count = bytes.len().min(2);
        self.0.extend_from_slice(&bytes[..count]);
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct VectoredWriter {
    bytes: Vec<u8>,
    outcomes: VecDeque<Result<usize, io::ErrorKind>>,
}
impl Write for VectoredWriter {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        panic!("expected vectored writes")
    }
    fn write_vectored(&mut self, slices: &[IoSlice<'_>]) -> io::Result<usize> {
        let limit = self
            .outcomes
            .pop_front()
            .unwrap_or(Ok(usize::MAX))
            .map_err(io::Error::from)?;
        let mut count = 0;
        for slice in slices {
            let end = slice.len().min(limit - count);
            self.bytes.extend_from_slice(&slice[..end]);
            count += end;
            if count == limit {
                break;
            }
        }
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn vectored_writes_retry_interruptions_and_advance_partial_slices() {
    let mut writer = VectoredWriter {
        bytes: Vec::new(),
        outcomes: [Err(io::ErrorKind::Interrupted), Ok(1), Ok(6), Ok(2)].into(),
    };
    let mut slices = [
        IoSlice::new(b""),
        IoSlice::new(b"abc"),
        IoSlice::new(b"defg"),
        IoSlice::new(b""),
        IoSlice::new(b"hijk"),
    ];
    write_all_slices(&mut writer, &mut slices).unwrap();
    assert_eq!(writer.bytes, b"abcdefghijk");
}

#[test]
fn vectored_writes_support_scalar_fallback_and_empty_buffers() {
    let mut writer = ScalarWriter(Vec::new());
    let mut slices = [
        IoSlice::new(b""),
        IoSlice::new(b"abc"),
        IoSlice::new(b""),
        IoSlice::new(b"def"),
        IoSlice::new(b""),
    ];
    write_all_slices(&mut writer, &mut slices).unwrap();
    assert_eq!(writer.0, b"abcdef");
    write_all_slices(&mut writer, &mut []).unwrap();
    write_all_slices(&mut writer, &mut [IoSlice::new(b"")]).unwrap();
}

#[test]
fn vectored_writes_report_zero_progress_and_propagate_errors() {
    for outcome in [Ok(0), Err(io::ErrorKind::PermissionDenied)] {
        let expected = match outcome {
            Ok(_) => io::ErrorKind::WriteZero,
            Err(kind) => kind,
        };
        let mut writer = VectoredWriter {
            bytes: Vec::new(),
            outcomes: [Ok(1), outcome].into(),
        };
        let error = write_all_slices(&mut writer, &mut [IoSlice::new(b"abc")]).unwrap_err();
        assert_eq!(error.kind(), expected);
        assert_eq!(writer.bytes, b"a");
    }
}
