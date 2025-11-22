import { Mail as LucideMail, LucideProps } from 'lucide-react';

const Envelope = ({ className, ...props }: LucideProps) => {
  return <LucideMail className={className} {...props} />;
};

export default Envelope;